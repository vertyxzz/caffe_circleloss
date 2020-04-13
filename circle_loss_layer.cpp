#include "caffe/layers/circle_loss_layer.hpp"
#include "math.h"

namespace caffe {

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
	
	batch_size_ = bottom[0]->num();
	fea_dim_ = bottom[0]->count(1);
	
	inner_matrix_.Reshape(batch_size_, batch_size_, 1, 1);
	norm_.Reshape(batch_size_, 1, 1, 1);
	bottom_diff_.Reshape(batch_size_, fea_dim_, 1, 1);
  diff_1_.Reshape(fea_dim_, 1, 1, 1);
  diff_2_.Reshape(fea_dim_, 1, 1, 1);
  
  int max_pair_num = batch_size_ * (batch_size_-1) / 2;
  logit_n_.Reshape(max_pair_num, 1, 1, 1);
  logit_p_.Reshape(max_pair_num, 1, 1, 1);
	prob_n_.Reshape(max_pair_num, 1, 1, 1);
  prob_p_.Reshape(max_pair_num, 1, 1, 1);
  cos_n_.Reshape(max_pair_num, 1, 1, 1);
  cos_p_.Reshape(max_pair_num, 1, 1, 1);
  inner_n_.Reshape(max_pair_num, 1, 1, 1);
  inner_p_.Reshape(max_pair_num, 1, 1, 1);
  idx_n1_.Reshape(max_pair_num, 1, 1, 1);
  idx_n2_.Reshape(max_pair_num, 1, 1, 1);
  idx_p1_.Reshape(max_pair_num, 1, 1, 1);
  idx_p2_.Reshape(max_pair_num, 1, 1, 1);
  
	gamma_ = this->layer_param_.circle_loss_param().gamma();
	margin_ = this->layer_param_.circle_loss_param().margin();
	delta_p_ = 1 - margin_;
	delta_n_ = margin_;
	optimum_p_ = 1 + margin_;
	optimum_n_ = -margin_;
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
	vector<int> loss_shape(0);
	top[0]->Reshape(loss_shape);
	caffe_set(bottom_diff_.count(), Dtype(0), bottom_diff_.template mutable_cpu_data<Dtype>());	
}

template <typename Ftype, typename Btype>
Ftype CircleLossLayer<Ftype, Btype>::calc_softplus(Dtype x) {
	// log(1+e(x)) = log(1+e(-abs(x))) + max(x,0)
	Dtype x_abs = x;
	Dtype x_max = x;
	if (x < 0) {
		x_abs *= -1;
		x_max = 0;
	}
	
	return log(1+exp(-x_abs)) + x_max;
}

template <typename Ftype, typename Btype>
Ftype CircleLossLayer<Ftype, Btype>::calc_logsumexp(const Dtype* x, int x_len, Dtype* prob) {
	// log(exp(x1)+exp(x2)+...) = log(exp(x1-max(x))+exp(x2-max(x))+...) + max(x)
	if (x_len <= 0)
		return 0;
	
	Dtype x_max = x[0];
	for (int i=1; i<x_len; ++i) {
		if (x_max < x[i])
			x_max = x[i];
	}
		
	Dtype sumexp = 0;
	for (int i=0; i<x_len; ++i) {
		prob[i] = exp(x[i] - x_max);
	  sumexp += prob[i];
	}
	
	if (prob) {
		Dtype scale = 1 / (sumexp + Dtype(1e-5));
		for (int i=0; i<x_len; ++i)
		  prob[i] *= scale;
	}
	
	Dtype logsumexp = log(sumexp) + x_max;	
	return logsumexp;   
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::ComputeDiff_cpu(const Dtype* x_1, const Dtype* x_2, 
    const Dtype x_1_norm, const Dtype x_2_norm,
    const Dtype inner_val, Dtype* x_1_diff) {
	caffe_cpu_scale<Dtype>(fea_dim_, Dtype(1) / (x_1_norm * x_2_norm), x_2, x_1_diff);
	Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
	caffe_cpu_axpby<Dtype>(fea_dim_, -inner_val / (x_1_norm_cubic * x_2_norm), x_1, Dtype(1), x_1_diff);
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
	const Dtype eps = Dtype(1e-5);  
  const Dtype* fea_val = bottom[0]->cpu_data<Dtype>();
  const Dtype* label_val = bottom[1]->cpu_data<Dtype>();
  Dtype* bottom_diff = bottom_diff_.template mutable_cpu_data<Dtype>();
  		
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, batch_size_, 
		fea_dim_, Dtype(1), fea_val, fea_val, Dtype(0), inner_matrix_.template mutable_cpu_data<Dtype>());
	
	const Dtype* inner_matrix_val = inner_matrix_.template cpu_data<Dtype>();
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
	  	}
	  }
	}
  
  Dtype* prob_n = prob_n_.template mutable_cpu_data<Dtype>();
	Dtype* prob_p = prob_p_.template mutable_cpu_data<Dtype>();
  Dtype lse_n = calc_logsumexp((const Dtype*)logit_n, n_num, prob_n);
  Dtype lse_p = calc_logsumexp((const Dtype*)logit_p, p_num, prob_p);
    
	Dtype loss = calc_softplus(lse_p + lse_n);
	top[0]->mutable_cpu_data<Dtype>()[0] = loss;	

	Dtype Z = 1 - exp(-loss);
	for (int i=0; i<n_num; ++i) {
	  Dtype loss_2_simi = Z * prob_n[i] * (gamma_ * (cos_n[i] + margin_));
	  int idx_1 = idx_n1[i];
	  int idx_2 = idx_n2[i];
	  const Dtype* fea_1 = fea_val + idx_1 * fea_dim_;
		const Dtype* fea_2 = fea_val + idx_2 * fea_dim_;
	  ComputeDiff_cpu(fea_1, fea_2, norm_data[idx_1], norm_data[idx_2], inner_n[i], diff_1_.template mutable_cpu_data<Dtype>());
	  ComputeDiff_cpu(fea_2, fea_1, norm_data[idx_2], norm_data[idx_1], inner_n[i], diff_2_.template mutable_cpu_data<Dtype>());

	  caffe_cpu_axpby(fea_dim_, loss_2_simi, diff_1_.template cpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_1 * fea_dim_));
	  caffe_cpu_axpby(fea_dim_, loss_2_simi, diff_2_.template cpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_2 * fea_dim_));
	}
	
	for (int i=0; i<p_num; ++i) {
	  Dtype loss_2_simi = Z * prob_p[i] * (gamma_ * (cos_p[i] - 1 - margin_));
	  int idx_1 = idx_p1[i];
	  int idx_2 = idx_p2[i];
	  const Dtype* fea_1 = fea_val + idx_1 * fea_dim_;
		const Dtype* fea_2 = fea_val + idx_2 * fea_dim_;
	  ComputeDiff_cpu(fea_1, fea_2, norm_data[idx_1], norm_data[idx_2], inner_p[i], diff_1_.template mutable_cpu_data<Dtype>());
	  ComputeDiff_cpu(fea_2, fea_1, norm_data[idx_2], norm_data[idx_1], inner_p[i], diff_2_.template mutable_cpu_data<Dtype>());

	  caffe_cpu_axpby(fea_dim_, loss_2_simi, diff_1_.template cpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_1 * fea_dim_));
	  caffe_cpu_axpby(fea_dim_, loss_2_simi, diff_2_.template cpu_data<Dtype>(), Dtype(1), bottom_diff + (idx_2 * fea_dim_));
	}
	
	return;
}

template <typename Ftype, typename Btype>
void CircleLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
	if (propagate_down[0]) {
	  Dtype scalar = top[0]->cpu_diff<Dtype>()[0] / (batch_size_-1);
		caffe_cpu_scale(bottom_diff_.count(), scalar, bottom_diff_.template cpu_data<Dtype>(), bottom[0]->mutable_cpu_diff<Dtype>());
	}
	
	return;
}

#ifdef CPU_ONLY
STUB_GPU(CircleLossLayer);
#endif

INSTANTIATE_CLASS_FB(CircleLossLayer);
REGISTER_LAYER_CLASS(CircleLoss);

}
