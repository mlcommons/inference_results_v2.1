#include <chrono>
#include <vector>
#include <string>
#include "omp.h"
//#include <pair>

using namespace std::chrono;
#include "backend.hpp"

Backend::Backend( std::string rn50_part1, std::string rn50_part3, std::string full_model, int batch_size) :
        rn50_start_path_(rn50_part1), rn50_end_path_(rn50_part3), full_model_path_(full_model), batch_size_(batch_size){
    if (batch_size==9) {
        this->rn50_backbone_ = &rn50_backbone_bs9;
        this->init_backbone_ = &sc_init_rn50_backbone_bs9;
    }
     else if (batch_size==4) {
        this->rn50_backbone_ = &rn50_backbone_bs4;
        this->init_backbone_ = &sc_init_rn50_backbone_bs4;
    }
     else if (batch_size==2) {
        this->rn50_backbone_ = &rn50_backbone_bs2;
        this->init_backbone_ = &sc_init_rn50_backbone_bs2;
    }
}

void Backend::prepareOneDNN() {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    dnnl::memory::dims conv_src_tz = {this->batch_size_, Start_In_C, Start_I_H, Start_I_W};
    dnnl::memory::dims conv_weights_tz = {Start_Out_C, Start_In_C, Start_W_H, Start_W_W};
    dnnl::memory::dims conv_bias_tz = {Start_Out_C};
    dnnl::memory::dims conv_dst_tz = {this->batch_size_, Start_Out_C, Start_Out_H, Start_Out_W};
    dnnl::memory::dims conv_strides = {2, 2};
    dnnl::memory::dims conv_padding = {3, 3};

    auto user_weights_memory = dnnl::memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng_);
    memcpy(user_weights_memory.get_data_handle(), weight_map_["conv1.weight"].data_ptr<float>(), Start_Out_C*Start_In_C*Start_W_H*Start_W_W*sizeof(float));
    auto user_bias_memory = dnnl::memory({{conv_bias_tz}, dt::f32, tag::x}, eng_);
    memcpy(user_bias_memory.get_data_handle(), weight_map_["conv1.bias"].data_ptr<float>(), Start_Out_C*sizeof(float));

    const float post_scale = 1.f;

    const std::vector<float> weight_scales = {post_scale/0.0006816235836595297f,
                                            post_scale/0.0005305774975568056f,
                                            post_scale/0.00016325304750353098f,
                                            post_scale/0.0002625430643092841f,
                                            post_scale/0.001501173130236566f,
                                            post_scale/0.0002946545137092471f,
                                            post_scale/0.0013540860963985324f,
                                            post_scale/0.0012125875800848007f,
                                            post_scale/0.0018035992980003357f,
                                            post_scale/0.00025723729049786925f,
                                            post_scale/0.001094487844966352f,
                                            post_scale/0.00026009505381807685f,
                                            post_scale/0.0008863380062393844f,
                                            post_scale/1.1920928955078125e-07f,
                                            post_scale/0.0024081964511424303f,
                                            post_scale/0.0015865263994783163f,
                                            post_scale/0.0004841082845814526f,
                                            post_scale/0.001560483011417091f,
                                            post_scale/0.0025746244937181473f,
                                            post_scale/0.000310599833028391f,
                                            post_scale/0.002417005831375718f,
                                            post_scale/0.0015911447117105126f,
                                            post_scale/0.0022614472545683384f,
                                            post_scale/0.00043826879118569195f,
                                            post_scale/0.0007691492792218924f,
                                            post_scale/0.00020982741261832416f,
                                            post_scale/0.0012938278960064054f,
                                            post_scale/0.0012921467423439026f,
                                            post_scale/0.0004248563782311976f,
                                            post_scale/0.00024047742772381753f,
                                            post_scale/0.0005666522774845362f,
                                            post_scale/0.00013184541603550315f,
                                            post_scale/0.0002759526832960546f,
                                            post_scale/0.0038828933611512184f,
                                            post_scale/0.00014129547344055027f,
                                            post_scale/0.00030801387038081884f,
                                            post_scale/0.0002745217061601579f,
                                            post_scale/0.00030328892171382904f,
                                            post_scale/0.0010830751853063703f,
                                            post_scale/0.0011779471533372998f,
                                            post_scale/0.00022047704260330647f,
                                            post_scale/0.001532222144305706f,
                                            post_scale/0.0002601104788482189f,
                                            post_scale/0.00037946386146359146f,
                                            post_scale/0.0001750480878399685f,
                                            post_scale/0.0008486405131407082f,
                                            post_scale/0.0015950539382174611f,
                                            post_scale/0.0004155198694206774f,
                                            post_scale/0.0008059836691245437f,
                                            post_scale/0.0017546059098094702f,
                                            post_scale/0.001436483347788453f,
                                            post_scale/0.000660675170365721f,
                                            post_scale/0.00042748029227368534f,
                                            post_scale/0.0012573313433676958f,
                                            post_scale/0.0011386079713702202f,
                                            post_scale/0.000577236816752702f,
                                            post_scale/0.0005883735720999539f,
                                            post_scale/0.00015980478201527148f,
                                            post_scale/0.00045653333654627204f,
                                            post_scale/0.00031122786458581686f,
                                            post_scale/0.0006546518416143954f,
                                            post_scale/0.0009724263800308108f,
                                            post_scale/0.00027550148661248386f,
                                            post_scale/0.0025447772350162268f};

    const float in_scale = 0.02070588245987892;
    const std::vector<float> conv_scales = {in_scale*0.0006816235836595297f,
                                            in_scale*0.0005305774975568056f,
                                            in_scale*0.00016325304750353098f,
                                            in_scale*0.0002625430643092841f,
                                            in_scale*0.001501173130236566f,
                                            in_scale*0.0002946545137092471f,
                                            in_scale*0.0013540860963985324f,
                                            in_scale*0.0012125875800848007f,
                                            in_scale*0.0018035992980003357f,
                                            in_scale*0.00025723729049786925f,
                                            in_scale*0.001094487844966352f,
                                            in_scale*0.00026009505381807685f,
                                            in_scale*0.0008863380062393844f,
                                            in_scale*1.1920928955078125e-07f,
                                            in_scale*0.0024081964511424303f,
                                            in_scale*0.0015865263994783163f,
                                            in_scale*0.0004841082845814526f,
                                            in_scale*0.001560483011417091f,
                                            in_scale*0.0025746244937181473f,
                                            in_scale*0.000310599833028391f,
                                            in_scale*0.002417005831375718f,
                                            in_scale*0.0015911447117105126f,
                                            in_scale*0.0022614472545683384f,
                                            in_scale*0.00043826879118569195f,
                                            in_scale*0.0007691492792218924f,
                                            in_scale*0.00020982741261832416f,
                                            in_scale*0.0012938278960064054f,
                                            in_scale*0.0012921467423439026f,
                                            in_scale*0.0004248563782311976f,
                                            in_scale*0.00024047742772381753f,
                                            in_scale*0.0005666522774845362f,
                                            in_scale*0.00013184541603550315f,
                                            in_scale*0.0002759526832960546f,
                                            in_scale*0.0038828933611512184f,
                                            in_scale*0.00014129547344055027f,
                                            in_scale*0.00030801387038081884f,
                                            in_scale*0.0002745217061601579f,
                                            in_scale*0.00030328892171382904f,
                                            in_scale*0.0010830751853063703f,
                                            in_scale*0.0011779471533372998f,
                                            in_scale*0.00022047704260330647f,
                                            in_scale*0.001532222144305706f,
                                            in_scale*0.0002601104788482189f,
                                            in_scale*0.00037946386146359146f,
                                            in_scale*0.0001750480878399685f,
                                            in_scale*0.0008486405131407082f,
                                            in_scale*0.0015950539382174611f,
                                            in_scale*0.0004155198694206774f,
                                            in_scale*0.0008059836691245437f,
                                            in_scale*0.0017546059098094702f,
                                            in_scale*0.001436483347788453f,
                                            in_scale*0.000660675170365721f,
                                            in_scale*0.00042748029227368534f,
                                            in_scale*0.0012573313433676958f,
                                            in_scale*0.0011386079713702202f,
                                            in_scale*0.000577236816752702f,
                                            in_scale*0.0005883735720999539f,
                                            in_scale*0.00015980478201527148f,
                                            in_scale*0.00045653333654627204f,
                                            in_scale*0.00031122786458581686f,
                                            in_scale*0.0006546518416143954f,
                                            in_scale*0.0009724263800308108f,
                                            in_scale*0.00027550148661248386f,
                                            in_scale*0.0025447772350162268f};

    const std::vector<float> bias_scales = {1/(in_scale*0.0006816235836595297f),
                                            1/(in_scale*0.0005305774975568056f),
                                            1/(in_scale*0.00016325304750353098f),
                                            1/(in_scale*0.0002625430643092841f),
                                            1/(in_scale*0.001501173130236566f),
                                            1/(in_scale*0.0002946545137092471f),
                                            1/(in_scale*0.0013540860963985324f),
                                            1/(in_scale*0.0012125875800848007f),
                                            1/(in_scale*0.0018035992980003357f),
                                            1/(in_scale*0.00025723729049786925f),
                                            1/(in_scale*0.001094487844966352f),
                                            1/(in_scale*0.00026009505381807685f),
                                            1/(in_scale*0.0008863380062393844f),
                                            1/(in_scale*1.1920928955078125e-07f),
                                            1/(in_scale*0.0024081964511424303f),
                                            1/(in_scale*0.0015865263994783163f),
                                            1/(in_scale*0.0004841082845814526f),
                                            1/(in_scale*0.001560483011417091f),
                                            1/(in_scale*0.0025746244937181473f),
                                            1/(in_scale*0.000310599833028391f),
                                            1/(in_scale*0.002417005831375718f),
                                            1/(in_scale*0.0015911447117105126f),
                                            1/(in_scale*0.0022614472545683384f),
                                            1/(in_scale*0.00043826879118569195f),
                                            1/(in_scale*0.0007691492792218924f),
                                            1/(in_scale*0.00020982741261832416f),
                                            1/(in_scale*0.0012938278960064054f),
                                            1/(in_scale*0.0012921467423439026f),
                                            1/(in_scale*0.0004248563782311976f),
                                            1/(in_scale*0.00024047742772381753f),
                                            1/(in_scale*0.0005666522774845362f),
                                            1/(in_scale*0.00013184541603550315f),
                                            1/(in_scale*0.0002759526832960546f),
                                            1/(in_scale*0.0038828933611512184f),
                                            1/(in_scale*0.00014129547344055027f),
                                            1/(in_scale*0.00030801387038081884f),
                                            1/(in_scale*0.0002745217061601579f),
                                            1/(in_scale*0.00030328892171382904f),
                                            1/(in_scale*0.0010830751853063703f),
                                            1/(in_scale*0.0011779471533372998f),
                                            1/(in_scale*0.00022047704260330647f),
                                            1/(in_scale*0.001532222144305706f),
                                            1/(in_scale*0.0002601104788482189f),
                                            1/(in_scale*0.00037946386146359146f),
                                            1/(in_scale*0.0001750480878399685f),
                                            1/(in_scale*0.0008486405131407082f),
                                            1/(in_scale*0.0015950539382174611f),
                                            1/(in_scale*0.0004155198694206774f),
                                            1/(in_scale*0.0008059836691245437f),
                                            1/(in_scale*0.0017546059098094702f),
                                            1/(in_scale*0.001436483347788453f),
                                            1/(in_scale*0.000660675170365721f),
                                            1/(in_scale*0.00042748029227368534f),
                                            1/(in_scale*0.0012573313433676958f),
                                            1/(in_scale*0.0011386079713702202f),
                                            1/(in_scale*0.000577236816752702f),
                                            1/(in_scale*0.0005883735720999539f),
                                            1/(in_scale*0.00015980478201527148f),
                                            1/(in_scale*0.00045653333654627204f),
                                            1/(in_scale*0.00031122786458581686f),
                                            1/(in_scale*0.0006546518416143954f),
                                            1/(in_scale*0.0009724263800308108f),
                                            1/(in_scale*0.00027550148661248386f),
                                            1/(in_scale*0.0025447772350162268f)};
    // std::vector<float> conv_scales = {0.5f};

    const int weight_mask = 1;
    const int bias_mask = 1;
    const int conv_mask = 2;

    auto conv_weights_memory = dnnl::memory({{conv_weights_tz}, dt::s8, tag::Adcb16a}, eng_);
    dnnl::primitive_attr weight_attr;
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd = dnnl::reorder::primitive_desc(eng_, user_weights_memory.get_desc(),
                                                           eng_, conv_weights_memory.get_desc(), weight_attr);
    auto weight_reorder = dnnl::reorder(weight_reorder_pd);
    weight_reorder.execute(s_, user_weights_memory, conv_weights_memory);

    // float* weight_ptr = (float*) user_weights_memory.get_data_handle();
    // printf("FP weights %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", weight_ptr[0], weight_ptr[1], weight_ptr[2], weight_ptr[3], weight_ptr[4], weight_ptr[5], weight_ptr[6], weight_ptr[7], weight_ptr[8], weight_ptr[9], weight_ptr[10], weight_ptr[11]);
    // int8_t* input_ptr = (int8_t*) conv_weights_memory.get_data_handle();
    // printf("Quantized weights %d, %d, %d, %d\n", input_ptr[0], input_ptr[1], input_ptr[2], input_ptr[3]);

    auto conv_bias_memory = dnnl::memory({{conv_bias_tz}, dt::f32, tag::a}, eng_);
    dnnl::primitive_attr bias_attr;
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd = dnnl::reorder::primitive_desc(eng_, user_bias_memory.get_desc(),
                                                         eng_, conv_bias_memory.get_desc(), bias_attr);
    auto bias_reorder = dnnl::reorder(bias_reorder_pd);
    bias_reorder.execute(s_, user_bias_memory, conv_bias_memory);

    // float* conv_bias_ptr = (float*) conv_bias_memory.get_data_handle();
    // std::cout << "bias " << conv_bias_ptr[0] << " " << conv_bias_ptr[1] <<std::endl;

    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::s8, tag::any);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::s8, tag::any);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::s8, tag::any);

    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md, 
            conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);
    
    dnnl::primitive_attr conv_attr;
    conv_attr.set_output_scales(conv_mask, conv_scales);

    const float ops_scale = 1./0.05720944702625275;
    const float ops_alpha = 0.f; // SKip?
    const float ops_beta = 0.f;
    dnnl::post_ops ops;
    ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, conv_attr, eng_);

    auto conv_dst_memory = dnnl::memory(conv_prim_desc.dst_desc(), eng_);

    net_.push_back(dnnl::convolution_forward(conv_prim_desc));
    net_args_.push_back({{DNNL_ARG_SRC, conv_src_memory_},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_bias_memory},
                        {DNNL_ARG_DST, conv_dst_memory}});

    dnnl::memory::dims pool_dst_tz = {this->batch_size_, 64, 56, 56};
    dnnl::memory::dims pool_kernel = {3,3};
    dnnl::memory::dims pool_strides = {2,2};
    dnnl::memory::dims pool_padding = {1,1};

    auto pool_dst_md = dnnl::memory::desc({pool_dst_tz}, dt::s8, tag::any);
    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_max, conv_dst_memory.get_desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, eng_);
    
    net_.push_back(dnnl::pooling_forward(pool_pd));
    net_args_.push_back({{DNNL_ARG_SRC, conv_dst_memory},
                         {DNNL_ARG_DST, pool_dst_memory_}});
}

void Backend::runStart() {
    for (size_t i=0; i<net_.size(); ++i) {
        net_.at(i).execute(s_, net_args_.at(i));
    }
}

void Backend::load(){

    // Load first model
    model_start_ = torch::jit::load(rn50_start_path_);
    model_start_.eval();

    // Initialize custom kernels
    init_backbone_();

    //Load final model
    model_end_ = torch::jit::load(rn50_end_path_);
    model_end_.eval();

    // Load fp weights of "backbone"
    setWeightMap();

    // Prepare OneDNN handles
    prepareOneDNN();
}

torch::jit::IValue Backend::predict(torch::Tensor& input_tensor, torch::Tensor& backbone_output){
    //------- First stage inference -------
    this->runStart();
    
    // start_out_ = model_start_.forward({input_tensor}).toTensor();
    // backbone_in_ptr_ = start_out_.data_ptr<int8_t>();
    
    // printf("%i, %i, %i, %i, %i, %i, %i, %i\n", 
    // backbone_in_ptr_[0], backbone_in_ptr_[1], backbone_in_ptr_[2], backbone_in_ptr_[3],
    // backbone_in_ptr_[4], backbone_in_ptr_[5], backbone_in_ptr_[6], backbone_in_ptr_[7]);
    int8_t* backbone_out_ptr = backbone_output.data_ptr<int8_t>();

    runBackboneKernel(backbone_out_ptr, backbone_in_ptr_);
   
    auto end_out = this->model_end_({backbone_output});
    return end_out;
}

void Backend::setWeightMap(){
    torch::jit::script::Module traced_fp_model = torch::jit::load(full_model_path_);
    for (const auto& pair : traced_fp_model.named_parameters()) {
        std::string pname = pair.name;
        torch::Tensor value = pair.value;
        weight_map_.insert(std::pair<std::string, at::Tensor>{pname, value});
    }
}
