# Rank-1 NAS Model — 7 Ports

## Selection

- Rank: 1
- Trial: 38
- Ranking criterion: lowest quantized validation outage probability (best_qt_val_metric)
- Quantized validation outage probability: 0.3077992857142857
- FP32 validation outage probability: 0.30181714285714284
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 32
- Learning rate: 0.0001318104936913322
- Ranking source: results/nas_pqt_mobv3_all_regimes/nas_analysis/nas_ptq_mobv3_all_regimes_7_ports/tables/top_k_trials.csv

## Artifacts and Model Size

| Artifact | Format | Exact size | Size (KiB) | SHA-256 |
|---|---|---:|---:|---|
| top_1_trial_38.keras | Keras v3 archive, FP32 weights | 243663 bytes | 237.952148 KiB | 20bda7fd996eb7711ca6228d9eb00142d85f7650ae05a5ecbaf136ff1fa2e86a |
| top_1_trial_38.tflite | TensorFlow Lite, strict INT8 | 158312 bytes | 154.601562 KiB | 4d9cb5161b667ce9ca492cffa752886729190ccb9339ac56f9aec40f0724c00a |

The sizes above are measured filesystem sizes of the copied artifacts, not estimates.

## Parameters

| Parameter category | Count | FP32 tensor storage |
|---|---:|---:|
| Trainable | 15,756 | 63,024 bytes |
| Non-trainable | 0 | 0 bytes |
| Total | 15,756 | 63,024 bytes |

Parameter storage is parameter count multiplied by 4 bytes for the FP32 Keras weights. Archive size differs because the Keras file also stores graph metadata and container overhead. The INT8 TFLite file includes quantized tensors plus its FlatBuffer graph.

## Exact Serialized Keras Structure

- Model name: mobv3_alpha02_gn_safe_selector_protocol
- Input: shape (None, 32, 32, 3), dtype float32
- Output: shape (None, 100), dtype float32
- The table expands every nested Functional model. A Functional row is a container subtotal; its parameters are also represented by its nested parameterized layers and must not be added again.

| # | Exact scoped layer name | Type | Inbound layer(s) | Input shape(s) | Output shape(s) | Parameters | Exact structural configuration |
|---:|---|---|---|---|---|---:|---|
| 1 | mobv3_alpha02_gn_safe_selector_protocol/sparse_port_image | InputLayer | — | [] | (None, 32, 32, 3) | 0 | {"batch_shape":[null,32,32,3],"dtype":"float32"} |
| 2 | mobv3_alpha02_gn_safe_selector_protocol/sparse_image_observed_values | SparseImageObservedValues | sparse_port_image | (None, 32, 32, 3) | (None, 7) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,16,33,49,66,82,99]} |
| 3 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector | Functional | sparse_image_observed_values | (None, 7) | (None, 100) | 15,756 | {"nested_layers":50} |
| 4 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/observed_snr | InputLayer | — | [] | (None, 7) | 0 | {"batch_shape":[null,7],"dtype":"float32"} |
| 5 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/expanded_port_image | ExpandedPortImage | observed_snr | (None, 7) | (None, 32, 32, 3) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,16,33,49,66,82,99],"log_scale":3.018411636352539,"image_side":32} |
| 6 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/nas_stem_conv | Conv2D | expanded_port_image | (None, 32, 32, 3) | (None, 16, 16, 16) | 432 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[3,3],"strides":[2,2],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 7 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/nas_stem_gn | GroupNormalization | nas_stem_conv | (None, 16, 16, 16) | (None, 16, 16, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 8 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/nas_stem_activation | Activation | nas_stem_gn | (None, 16, 16, 16) | (None, 16, 16, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"hard_silu"} |
| 9 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_expand | Conv2D | nas_stem_activation | (None, 16, 16, 16) | (None, 16, 16, 16) | 256 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 10 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_expand_gn | GroupNormalization | v3_block_0_expand | (None, 16, 16, 16) | (None, 16, 16, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 11 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_expand_activation | Activation | v3_block_0_expand_gn | (None, 16, 16, 16) | (None, 16, 16, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 12 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_depthwise | DepthwiseConv2D | v3_block_0_expand_activation | (None, 16, 16, 16) | (None, 8, 8, 16) | 144 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"kernel_size":[3,3],"strides":[2,2],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"depth_multiplier":1,"use_bias":false,"activation":"linear"} |
| 13 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_depthwise_gn | GroupNormalization | v3_block_0_depthwise | (None, 8, 8, 16) | (None, 8, 8, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 14 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_depthwise_activation | Activation | v3_block_0_depthwise_gn | (None, 8, 8, 16) | (None, 8, 8, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 15 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_se_pool | GlobalAveragePooling2D | v3_block_0_depthwise_activation | (None, 8, 8, 16) | (None, 1, 1, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"data_format":"channels_last","keepdims":true} |
| 16 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_se_reduce | Conv2D | v3_block_0_se_pool | (None, 1, 1, 16) | (None, 1, 1, 8) | 136 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":8,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":true,"activation":"relu"} |
| 17 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_se_expand | Conv2D | v3_block_0_se_reduce | (None, 1, 1, 8) | (None, 1, 1, 16) | 144 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":true,"activation":"hard_sigmoid"} |
| 18 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_se_scale | Multiply | v3_block_0_depthwise_activation, v3_block_0_se_expand | [(None, 8, 8, 16), (None, 1, 1, 16)] | (None, 8, 8, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null}} |
| 19 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_project | Conv2D | v3_block_0_se_scale | (None, 8, 8, 16) | (None, 8, 8, 16) | 256 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 20 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_0_project_gn | GroupNormalization | v3_block_0_project | (None, 8, 8, 16) | (None, 8, 8, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 21 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand | Conv2D | v3_block_0_project_gn | (None, 8, 8, 16) | (None, 8, 8, 40) | 640 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":40,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 22 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand_gn | GroupNormalization | v3_block_1_expand | (None, 8, 8, 40) | (None, 8, 8, 40) | 80 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 23 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand_activation | Activation | v3_block_1_expand_gn | (None, 8, 8, 40) | (None, 8, 8, 40) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 24 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise | DepthwiseConv2D | v3_block_1_expand_activation | (None, 8, 8, 40) | (None, 4, 4, 40) | 360 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"kernel_size":[3,3],"strides":[2,2],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"depth_multiplier":1,"use_bias":false,"activation":"linear"} |
| 25 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise_gn | GroupNormalization | v3_block_1_depthwise | (None, 4, 4, 40) | (None, 4, 4, 40) | 80 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 26 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise_activation | Activation | v3_block_1_depthwise_gn | (None, 4, 4, 40) | (None, 4, 4, 40) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 27 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_project | Conv2D | v3_block_1_depthwise_activation | (None, 4, 4, 40) | (None, 4, 4, 16) | 640 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 28 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_project_gn | GroupNormalization | v3_block_1_project | (None, 4, 4, 16) | (None, 4, 4, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 29 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_expand | Conv2D | v3_block_1_project_gn | (None, 4, 4, 16) | (None, 4, 4, 56) | 896 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":56,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 30 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_expand_gn | GroupNormalization | v3_block_2_expand | (None, 4, 4, 56) | (None, 4, 4, 56) | 112 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 31 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_expand_activation | Activation | v3_block_2_expand_gn | (None, 4, 4, 56) | (None, 4, 4, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 32 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_depthwise | DepthwiseConv2D | v3_block_2_expand_activation | (None, 4, 4, 56) | (None, 4, 4, 56) | 504 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"kernel_size":[3,3],"strides":[1,1],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"depth_multiplier":1,"use_bias":false,"activation":"linear"} |
| 33 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_depthwise_gn | GroupNormalization | v3_block_2_depthwise | (None, 4, 4, 56) | (None, 4, 4, 56) | 112 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 34 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_depthwise_activation | Activation | v3_block_2_depthwise_gn | (None, 4, 4, 56) | (None, 4, 4, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 35 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_project | Conv2D | v3_block_2_depthwise_activation | (None, 4, 4, 56) | (None, 4, 4, 16) | 896 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 36 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_project_gn | GroupNormalization | v3_block_2_project | (None, 4, 4, 16) | (None, 4, 4, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 37 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_2_add | Add | v3_block_1_project_gn, v3_block_2_project_gn | [(None, 4, 4, 16), (None, 4, 4, 16)] | (None, 4, 4, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null}} |
| 38 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_expand | Conv2D | v3_block_2_add | (None, 4, 4, 16) | (None, 4, 4, 56) | 896 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":56,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 39 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_expand_gn | GroupNormalization | v3_block_3_expand | (None, 4, 4, 56) | (None, 4, 4, 56) | 112 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 40 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_expand_activation | Activation | v3_block_3_expand_gn | (None, 4, 4, 56) | (None, 4, 4, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"hard_silu"} |
| 41 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_depthwise | DepthwiseConv2D | v3_block_3_expand_activation | (None, 4, 4, 56) | (None, 2, 2, 56) | 1,400 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"kernel_size":[5,5],"strides":[2,2],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"depth_multiplier":1,"use_bias":false,"activation":"linear"} |
| 42 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_depthwise_gn | GroupNormalization | v3_block_3_depthwise | (None, 2, 2, 56) | (None, 2, 2, 56) | 112 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 43 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_depthwise_activation | Activation | v3_block_3_depthwise_gn | (None, 2, 2, 56) | (None, 2, 2, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"hard_silu"} |
| 44 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_se_pool | GlobalAveragePooling2D | v3_block_3_depthwise_activation | (None, 2, 2, 56) | (None, 1, 1, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"data_format":"channels_last","keepdims":true} |
| 45 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_se_reduce | Conv2D | v3_block_3_se_pool | (None, 1, 1, 56) | (None, 1, 1, 16) | 912 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":true,"activation":"relu"} |
| 46 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_se_expand | Conv2D | v3_block_3_se_reduce | (None, 1, 1, 16) | (None, 1, 1, 56) | 952 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":56,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":true,"activation":"hard_sigmoid"} |
| 47 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_se_scale | Multiply | v3_block_3_depthwise_activation, v3_block_3_se_expand | [(None, 2, 2, 56), (None, 1, 1, 56)] | (None, 2, 2, 56) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null}} |
| 48 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_project | Conv2D | v3_block_3_se_scale | (None, 2, 2, 56) | (None, 2, 2, 24) | 1,344 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":24,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 49 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_3_project_gn | GroupNormalization | v3_block_3_project | (None, 2, 2, 24) | (None, 2, 2, 24) | 48 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 50 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/global_average_pool | GlobalAveragePooling2D | v3_block_3_project_gn | (None, 2, 2, 24) | (None, 24) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"data_format":"channels_last","keepdims":false} |
| 51 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/selector_dense_0 | Dense | global_average_pool | (None, 24) | (None, 32) | 800 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"use_bias":true,"activation":"relu","units":32} |
| 52 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/exploration_logits | Dense | selector_dense_0 | (None, 32) | (None, 100) | 3,300 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"use_bias":true,"activation":"linear","units":100} |
| 53 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/safe_decision_scores | SafeDecisionScores | observed_snr, exploration_logits | [(None, 7), (None, 100)] | (None, 100) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,16,33,49,66,82,99],"threshold":1.1885398626327515,"total_ports":100} |

All 53 serialized layer/container rows are listed above. The authoritative unique parameter total is 15,756.

## TensorFlow Lite Export

| Property | Exact value |
|---|---|
| Input tensor | serving_default_sparse_port_image:0 |
| Input shape | (1, 32, 32, 3) |
| Input dtype | int8 |
| Input quantization (scale, zero_point) | (0.14461839199066162, -128) |
| Output tensor | StatefulPartitionedCall_1:0 |
| Output shape | (1, 100) |
| Output dtype | int8 |
| Output quantization (scale, zero_point) | (0.019607843831181526, -26) |
| Built-in TFLite operators | 302 |

The TFLite converter lowers, decomposes, and fuses Keras layers into operators. Therefore, the exact human-readable layer architecture is the serialized Keras graph above; the TFLite operator count is not a second layer count.
