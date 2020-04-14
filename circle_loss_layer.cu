#include "caffe/layers/circle_loss_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::ComputeDiff_gpu(const Dtype* x_1, const Dtype* x_2,
    const Dtype x_1_norm, const Dtype x_2_norm,
    const Dtype inner_val, Dtype* x_1_diff) {
  caffe_gpu_scale<Dtype>(fea_dim_, Dtype(1) / (x_1_norm * x_2_norm), x_2, x_1_diff);
  Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
  caffe_gpu_axpby<Dtype>(fea_dim_, -inner_val / (x_1_norm_cubic * x_2_norm), x_1, Dtype(1), x_1_diff);
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Dtype eps = Dtype(1e-5);
  const Dtype* fea_val = bottom[0]->gpu_data<Dtype>();
  const Dtype* label_val = bottom[1]->cpu_data<Dtype>();
  Dtype* bottom_diff = bottom_diff_.template mutable_gpu_data<Dtype>();
  caffe_set(bottom_diff_.count(), Dtype(0), bottom_diff_.template mutable_cpu_data<Dtype>());
  
//LOG(INFO) << "fea_1 = " << bottom[0]->cpu_data<Dtype>()[0] << ", fea_2 = " << bottom[0]->cpu_data<Dtype>()[1] << ", fea_3 = " << bottom[0]->cpu_data<Dtype>()[2];
//LOG(INFO) << "label_1 = " << label_val[0] << ", label_2 = " << label_val[1] << ", label_3 = " << label_val[2];

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, batch_size_,
    fea_dim_, Dtype(1), fea_val, fea_val, Dtype(0), inner_matrix_.template mutable_gpu_data<Dtype>());
    
  const Dtype* inner_matrix_val = inner_matrix_.template cpu_data<Dtype>();
  
//LOG(INFO) << "inner_val_1 = " << inner_matrix_val[0] << ", inner_val_2 = " << inner_matrix_val[1] << ", inner_val_3 = " << inner_matrix_val[2];

  Dtype* norm_data = norm_.template mutable_cpu_data<Dtype>();
  for (int i = 0; i < batch_size_; ++i)
    norm_data[i] = sqrt(inner_matrix_val[i * batch_size_ + i] + eps);
    
  Dtype* logit_n = logit_n_.template mutable_cpu_data<Dtype>();
  Dtype* logit_p = logit_p_.template mutable_cpu_data<Dtype>();
  
  Dtype* cos_n = cos_n_.template mutable_cpu_data<Dtype>();
  Dtype* cos_p = cos_p_.template mutable_cpu_data<Dtype>();
  Dtype* inner_n = inner_n_.template mutable_cpu_data<Dtype>();
  Dtype* inner_p = inner_p_.template mutable_cpu_data<Dtype>();
  
  Dtype* idx_n1 = idx_n1_.template mutable_cpu_data<Dtype>();
  Dtype* idx_n2 = idx_n2_.template mutable_cpu_data<Dtype>();
  Dtype* idx_p1 = idx_p1_.template mutable_cpu_data<Dtype>();
  Dtype* idx_p2 = idx_p2_.template mutable_cpu_data<Dtype>();
  
  int n_num = 0, p_num = 0;
  for (int i = 0; i < batch_size_-1; ++i) {
    int label1 = (int)label_val[i];
    for (int j = i+1; j < batch_size_; ++j) {
      int label2 = (int)label_val[j];
      int simi_pair = (label1 == label2) ? 1 : 0;
      
      Dtype inner_val = inner_matrix_val[i * batch_size_ + j];
      Dtype cos_sim = inner_val / (norm_data[i] * norm_data[j]);
      
      if (simi_pair) {
        //if (cos_sim < delta_p_)
        //  continue;
        
        Dtype alpha_p = optimum_p_ - cos_sim;
        if (alpha_p < 0)
          alpha_p = 0;
        cos_p[p_num] = cos_sim;
        inner_p[p_num] = inner_val;
        idx_p1[p_num] = i;
        idx_p2[p_num] = j;
        logit_p[p_num++] = (-1) * gamma_ * alpha_p * (cos_sim - delta_p_);
        
//LOG(INFO) << "i=" << i << ", j=" << j << ", cos = " << cos_sim << ", p_num=" << p_num << ", logit_p=" << logit_p[p_num-1];
      }
      else {
        //if (cos_sim > delta_n_)
        //  continue;
        
        Dtype alpha_n = cos_sim - optimum_n_;
        if (alpha_n < 0)
          alpha_n = 0;
        cos_n[n_num] = cos_sim;
        inner_n[n_num] = inner_val;
        idx_n1[n_num] = i;
        idx_n2[n_num] = j;
        logit_n[n_num++] = gamma_ * alpha_n * (cos_sim - delta_n_);
        
//LOG(INFO) << "i=" << i << ", j=" << j << ", cos = " << cos_sim << ", n_num=" << n_num << ", logit_n=" << logit_n[n_num-1];
      }
    }
  }
  
  Dtype* prob_n = prob_n_.template mutable_cpu_data<Dtype>();
  Dtype* prob_p = prob_p_.template mutable_cpu_data<Dtype>();
  Dtype lse_n = Safe_LogSumExp((const Dtype*)logit_n, n_num, prob_n);
  Dtype lse_p = Safe_LogSumExp((const Dtype*)logit_p, p_num, prob_p);
  
  //LOG(INFO) << "lse_p = " << lse_p << ", lse_n = " << lse_n << ", p_num = " << p_num << ", n_num = " << n_num;
  
  Dtype loss = Safe_SoftPlus(lse_p + lse_n);
  top[0]->mutable_cpu_data<Dtype>()[0] = loss;
  
  Dtype Z = 1 - exp(-loss);
  for (int i=0; i<n_num; ++i) {
    Dtype loss_2_simi = Z * prob_n[i] * (gamma_ * (cos_n[i] + margin_));
    int idx_1 = idx_n1[i];
    int idx_2 = idx_n2[i];
    const Dtype* fea_1 = fea_val + idx_1 * fea_dim_;
    const Dtype* fea_2 = fea_val + idx_2 * fea_dim_;
    ComputeDiff_gpu(fea_1, fea_2, norm_data[idx_1], norm_data[idx_2], inner_n[i], diff_1_.template mutable_gpu_data<Dtype>());
    ComputeDiff_gpu(fea_2, fea_1, norm_data[idx_2], norm_data[idx_1], inner_n[i], diff_2_.template mutable_gpu_data<Dtype>());
    
    caffe_gpu_axpby(fea_dim_, loss_2_simi, diff_1_.template gpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_1 * fea_dim_));
    caffe_gpu_axpby(fea_dim_, loss_2_simi, diff_2_.template gpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_2 * fea_dim_));
  }
  
  for (int i=0; i<p_num; ++i) {
    Dtype loss_2_simi = Z * prob_p[i] * (gamma_ * (cos_p[i] - 1 - margin_));
    int idx_1 = idx_p1[i];
    int idx_2 = idx_p2[i];
    const Dtype* fea_1 = fea_val + idx_1 * fea_dim_;
    const Dtype* fea_2 = fea_val + idx_2 * fea_dim_;
    ComputeDiff_gpu(fea_1, fea_2, norm_data[idx_1], norm_data[idx_2], inner_p[i], diff_1_.template mutable_gpu_data<Dtype>());
    ComputeDiff_gpu(fea_2, fea_1, norm_data[idx_2], norm_data[idx_1], inner_p[i], diff_2_.template mutable_gpu_data<Dtype>());
    
    caffe_gpu_axpby(fea_dim_, loss_2_simi, diff_1_.template gpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_1 * fea_dim_));
    caffe_gpu_axpby(fea_dim_, loss_2_simi, diff_2_.template gpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_2 * fea_dim_));
  }
  
  return;
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    Dtype scalar = top[0]->cpu_diff<Dtype>()[0] / (batch_size_-1);
    caffe_gpu_scale(bottom_diff_.count(), scalar, bottom_diff_.template gpu_data<Dtype>(), bottom[0]->mutable_gpu_diff<Dtype>());
  }
  
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CircleLossLayer);

}
