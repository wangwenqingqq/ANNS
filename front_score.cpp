// Offline all-vector RaBitQ adapter; no truth file or raw-vector reranking path.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace fs = std::filesystem;
using namespace rabitqlib;
struct Matrix { uint32_t n, d; std::vector<float> v; };
Matrix read_fbin(const fs::path& p) {
    std::ifstream f(p, std::ios::binary); Matrix m{};
    if (!f.read(reinterpret_cast<char*>(&m.n),4).read(reinterpret_cast<char*>(&m.d),4) ||
        !m.n || !m.d || fs::file_size(p)!=8+uint64_t(m.n)*m.d*4) throw std::runtime_error("Invalid fbin");
    m.v.resize(size_t(m.n)*m.d);
    if (!f.read(reinterpret_cast<char*>(m.v.data()), m.v.size()*4)) throw std::runtime_error("Truncated fbin");
    for (float v:m.v) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite input");
    return m;
}
template<class T> void write(const fs::path& p,const std::vector<T>& v) {
    std::ofstream f(p,std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T))) throw std::runtime_error("Output write failed");
}
int main(int argc,char** argv) {try {
    if (argc!=8) throw std::runtime_error("Usage: front_score base.fbin queries.fbin centers.fbin labels.u32 seed bits out_dir");
    auto start=std::chrono::steady_clock::now();
    Matrix x=read_fbin(argv[1]),q=read_fbin(argv[2]),c=read_fbin(argv[3]);
    const size_t n=x.n,dim=x.d,bits=std::stoul(argv[6]);
    if (q.d!=dim || c.d!=dim || !bits || bits>9) throw std::runtime_error("Shape/bits mismatch");
    fs::path out=argv[7]; if (!fs::create_directory(out)) throw std::runtime_error("Output exists; refusing overwrite");
    std::vector<uint32_t> labels(n);
    std::ifstream lf(argv[4],std::ios::binary);
    if (fs::file_size(argv[4])!=4*n || !lf.read(reinterpret_cast<char*>(labels.data()),4*n)) throw std::runtime_error("Invalid labels");
    std::unique_ptr<Rotator<float>> rot(choose_rotator<float>(dim));
    size_t pd=rot->size();
    std::mt19937 rng(static_cast<uint32_t>(std::stoul(argv[5])));
    std::uniform_int_distribution<int> flip(0,255);
    std::vector<char> state(rot->dump_bytes()); for (auto& v:state) v=static_cast<char>(flip(rng));
    rot->load(state.data()); write(out/"rotation.bin",state);
    std::vector<char> roundtrip(state.size());rot->save(roundtrip.data());
    if (roundtrip!=state) throw std::runtime_error("Rotation persistence mismatch");
    std::vector<float> centers(c.n*pd);for (size_t i=0;i<c.n;++i) rot->rotate(c.v.data()+i*dim,centers.data()+i*pd);
    write(out/"rotated_centers.f32",centers);
    std::vector<std::vector<uint32_t>> members(c.n);
    for (size_t i=0;i<n;++i) {if(labels[i]>=c.n)throw std::runtime_error("Invalid cluster");members[labels[i]].push_back(i);}
    std::vector<uint32_t> batch_centers,batch_ids;
    for(size_t k=0;k<c.n;++k) for(size_t i=0;i<members[k].size();i+=32) {
        batch_centers.push_back(k);
        for(size_t j=0;j<32;++j) batch_ids.push_back(i+j<members[k].size()?members[k][i+j]:UINT32_MAX);
    }
    const size_t batchbytes=BatchDataMap<float>::data_bytes(pd);
    const size_t exbytes=bits>1?ExDataMap<float>::data_bytes(pd,bits-1):0;
    std::vector<char> codes(batch_centers.size()*batchbytes),excodes(n*exbytes);
    quant::RabitqConfig config; // Exact enumeration, not the faster approximate encoder.
    double reference_max_error=0; size_t reference_checks=0;
    // Only encoding accesses raw database vectors. No raw x is used below the loop.
    #pragma omp parallel for schedule(dynamic)
    for(size_t b=0;b<batch_centers.size();++b) {
        std::vector<float> rotated(32*pd,0);size_t count=0;
        for(size_t j=0;j<32;++j) {auto id=batch_ids[b*32+j];if(id!=UINT32_MAX){rot->rotate(x.v.data()+id*dim,rotated.data()+j*pd);++count;}}
        auto cp=centers.data()+batch_centers[b]*pd;
        quant::quantize_one_batch(rotated.data(),cp,count,pd,codes.data()+b*batchbytes,METRIC_L2);
        if(bits>1)for(size_t j=0;j<count;++j)quant::quantize_compact_ex_bits(rotated.data()+j*pd,cp,pd,bits-1,excodes.data()+batch_ids[b*32+j]*exbytes,METRIC_L2,config);
    }
    write(out/"codes.bin",codes);write(out/"excodes.bin",excodes);write(out/"batch_ids.u32",batch_ids);write(out/"batch_centers.u32",batch_centers);
    // Independent packed-vs-unpacked check for the first vector of each cluster.
    std::vector<float> qr(pd);rot->rotate(q.v.data(),qr.data());
    for(size_t b=0;b<batch_centers.size();++b) {
        if(b && batch_centers[b]==batch_centers[b-1])continue;
        auto cp=centers.data()+batch_centers[b]*pd;
        SplitBatchQuery<float> query(qr.data(),pd,bits-1,METRIC_L2,true);
        query.set_g_add(std::sqrt(euclidean_sqr(qr.data(),cp,pd)));
        std::array<float,32> est,low,ip,ref,rl,ri;
        split_batch_estdist(codes.data()+b*batchbytes,query,pd,est.data(),low.data(),ip.data(),true);
        simd::split_batch_estdist_generic(codes.data()+b*batchbytes,query,pd,ref.data(),rl.data(),ri.data(),true);
        for(size_t j=0;j<32;++j)if(batch_ids[b*32+j]!=UINT32_MAX && std::abs(est[j]-ref[j])>2e-6*(1+std::abs(ref[j])))throw std::runtime_error("Dispatch/generic scorer mismatch");
        std::vector<float> xr(pd);rot->rotate(x.v.data()+batch_ids[b*32]*dim,xr.data());
        std::vector<uint32_t> unpacked(pd);float add,scale,error;
        quant::quantize_full_single(xr.data(),cp,pd,bits,unpacked.data(),add,scale,error,METRIC_L2,config);
        double dot=0, norm=0;const double cb=-(double(uint32_t(1)<<bits)-1)/2;
        for(size_t j=0;j<pd;++j){dot+=(double(unpacked[j])+cb)*double(qr[j]);norm+=std::pow(double(qr[j])-cp[j],2);}
        double expected=add+norm+scale*dot;
        double observed=est[0];
        if(bits>1)observed=split_distance_boosting(excodes.data()+batch_ids[b*32]*exbytes,select_excode_ipfunc(bits-1),query,pd,bits-1,ip[0]);
        double err=std::abs(observed-expected)/(1+norm+euclidean_sqr(xr.data(),cp,pd));
        reference_max_error=std::max(reference_max_error,err);++reference_checks;
        // HACC LUT quantization is not identical to the full-float reference.
        if(!std::isfinite(observed)||err>5e-4)throw std::runtime_error("Packed/scalar reference mismatch");
    }
    std::vector<float>().swap(x.v); // Diagnostic encoder scratch not a live scoring input.
    const double build_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream scores(out/"scores.fbin",std::ios::binary);
    uint32_t sn=q.n,sd=n;scores.write(reinterpret_cast<char*>(&sn),4).write(reinterpret_cast<char*>(&sd),4);
    auto exip=bits>1?select_excode_ipfunc(bits-1):nullptr;
    for(size_t qi=0;qi<q.n;++qi) {
        rot->rotate(q.v.data()+qi*dim,qr.data());
        std::vector<float> row(n);size_t count=0;
        SplitBatchQuery<float> query(qr.data(),pd,bits-1,METRIC_L2,true);
        for(size_t b=0;b<batch_centers.size();++b){
            auto cp=centers.data()+batch_centers[b]*pd;
            query.set_g_add(std::sqrt(euclidean_sqr(qr.data(),cp,pd)));
            std::array<float,32> est,low,ip;
            split_batch_estdist(codes.data()+b*batchbytes,query,pd,est.data(),low.data(),ip.data(),true);
            for(size_t j=0;j<32;++j){auto id=batch_ids[b*32+j];if(id==UINT32_MAX)continue;
                row[id]=bits==1?est[j]:split_distance_boosting(excodes.data()+id*exbytes,exip,query,pd,bits-1,ip[j]);
                if(!std::isfinite(row[id]))throw std::runtime_error("Nonfinite estimated distance");
                ++count;
            }
        }
        if(count!=n)throw std::runtime_error("Not all database vectors scored");
        if(!scores.write(reinterpret_cast<char*>(row.data()),n*4))throw std::runtime_error("Score output failed");
    }
    std::ofstream meta(out/"metadata.json");
    meta << "{\n\"bits\":"<<bits<<",\n\"n\":"<<n<<",\n\"query_n\":"<<q.n<<",\n\"padded_dim\":"<<pd
         <<",\n\"code_scored_count_per_query\":"<<n<<",\n\"raw_rerank\":false,\n\"hacc\":true,\n\"exact_encoder\":true"
         <<",\n\"code_bytes\":"<<codes.size()<<",\n\"excode_bytes\":"<<excodes.size()<<",\n\"batch_capacity\":"<<batch_ids.size()
         <<",\n\"persistent_payload_bytes\":"<<(codes.size()+excodes.size()+batch_ids.size()*4+batch_centers.size()*4+centers.size()*4+state.size())
         <<",\n\"vector_reserved_payload_bytes\":"<<(codes.capacity()+excodes.capacity()+batch_ids.capacity()*4+batch_centers.capacity()*4+centers.capacity()*4+state.capacity())
         <<",\n\"reference_checks\":"<<reference_checks<<",\n\"reference_max_normalized_error\":"<<reference_max_error
         <<",\n\"build_seconds\":"<<build_seconds<<",\n\"total_offline_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n}\n";
    if(!meta)throw std::runtime_error("Metadata write failed");
    std::cout<<"Scored "<<n<<" codes for each of "<<q.n<<" queries; bits="<<bits<<"; no raw rerank\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
