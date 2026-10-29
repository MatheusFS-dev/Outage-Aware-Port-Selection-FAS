# Rank-1 NAS Model — 4 Ports

## Selection

- Rank: 1
- Trial: 94
- Ranking criterion: lowest quantized validation outage probability (best_qt_val_metric)
- Quantized validation outage probability: 0.3237157142857143
- FP32 validation outage probability: 0.3146839285714286
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_1_project_gn
- Head depth: 1
- Head width: 64
- Learning rate: 0.00011044731521262944
- Ranking source: results/nas_pqt_mobv3_all_regimes/nas_analysis/nas_ptq_mobv3_all_regimes_4_ports/tables/top_k_trials.csv

## Artifacts and Model Size

| Artifact | Format | Exact size | Size (KiB) | SHA-256 |
|---|---|---:|---:|---|
| top_1_trial_94.keras | Keras v3 archive, FP32 weights | 155580 bytes | 151.933594 KiB | 5aca65a2803f6c2f8b3476da18b9bf0ad1e85b4bba58a2edf5a9f52e35719ce2 |
| top_1_trial_94.tflite | TensorFlow Lite, strict INT8 | 101400 bytes | 99.023438 KiB | 0959437f763ec6113711d66baf886aaf91e76765bc648d76e905ffddc13ee27b |

The sizes above are measured filesystem sizes of the copied artifacts, not estimates.

## Parameters

| Parameter category | Count | FP32 tensor storage |
|---|---:|---:|
| Trainable | 11,276 | 45,104 bytes |
| Non-trainable | 0 | 0 bytes |
| Total | 11,276 | 45,104 bytes |

Parameter storage is parameter count multiplied by 4 bytes for the FP32 Keras weights. Archive size differs because the Keras file also stores graph metadata and container overhead. The INT8 TFLite file includes quantized tensors plus its FlatBuffer graph.

## Exact Serialized Keras Structure

- Model name: mobv3_alpha02_gn_safe_selector_protocol
- Input: shape (None, 32, 32, 3), dtype float32
- Output: shape (None, 100), dtype float32
- The table expands every nested Functional model. A Functional row is a container subtotal; its parameters are also represented by its nested parameterized layers and must not be added again.

| # | Exact scoped layer name | Type | Inbound layer(s) | Input shape(s) | Output shape(s) | Parameters | Exact structural configuration |
|---:|---|---|---|---|---|---:|---|
| 1 | mobv3_alpha02_gn_safe_selector_protocol/sparse_port_image | InputLayer | — | [] | (None, 32, 32, 3) | 0 | {"batch_shape":[null,32,32,3],"dtype":"float32"} |
| 2 | mobv3_alpha02_gn_safe_selector_protocol/sparse_image_observed_values | SparseImageObservedValues | sparse_port_image | (None, 32, 32, 3) | (None, 4) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,33,66,99]} |
| 3 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector | Functional | sparse_image_observed_values | (None, 4) | (None, 100) | 11,276 | {"nested_layers":29} |
| 4 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/observed_snr | InputLayer | — | [] | (None, 4) | 0 | {"batch_shape":[null,4],"dtype":"float32"} |
| 5 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/expanded_port_image | ExpandedPortImage | observed_snr | (None, 4) | (None, 32, 32, 3) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,33,66,99],"log_scale":3.018411636352539,"image_side":32} |
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
| 21 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand | Conv2D | v3_block_0_project_gn | (None, 8, 8, 16) | (None, 8, 8, 48) | 768 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":48,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 22 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand_gn | GroupNormalization | v3_block_1_expand | (None, 8, 8, 48) | (None, 8, 8, 48) | 96 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 23 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_expand_activation | Activation | v3_block_1_expand_gn | (None, 8, 8, 48) | (None, 8, 8, 48) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 24 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise | DepthwiseConv2D | v3_block_1_expand_activation | (None, 8, 8, 48) | (None, 4, 4, 48) | 432 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"kernel_size":[3,3],"strides":[2,2],"padding":"same","data_format":"channels_last","dilation_rate":[1,1],"depth_multiplier":1,"use_bias":false,"activation":"linear"} |
| 25 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise_gn | GroupNormalization | v3_block_1_depthwise | (None, 4, 4, 48) | (None, 4, 4, 48) | 96 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 26 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_depthwise_activation | Activation | v3_block_1_depthwise_gn | (None, 4, 4, 48) | (None, 4, 4, 48) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"activation":"relu"} |
| 27 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_project | Conv2D | v3_block_1_depthwise_activation | (None, 4, 4, 48) | (None, 4, 4, 16) | 768 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"filters":16,"kernel_size":[1,1],"strides":[1,1],"padding":"valid","data_format":"channels_last","dilation_rate":[1,1],"groups":1,"use_bias":false,"activation":"linear"} |
| 28 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/v3_block_1_project_gn | GroupNormalization | v3_block_1_project | (None, 4, 4, 16) | (None, 4, 4, 16) | 32 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"groups":8,"axis":-1,"epsilon":0.001,"center":true,"scale":true} |
| 29 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/global_average_pool | GlobalAveragePooling2D | v3_block_1_project_gn | (None, 4, 4, 16) | (None, 16) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"data_format":"channels_last","keepdims":false} |
| 30 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/selector_dense_0 | Dense | global_average_pool | (None, 16) | (None, 64) | 1,088 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"use_bias":true,"activation":"relu","units":64} |
| 31 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/exploration_logits | Dense | selector_dense_0 | (None, 64) | (None, 100) | 6,500 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"use_bias":true,"activation":"linear","units":100} |
| 32 | mobv3_alpha02_gn_safe_selector_protocol/nas_ptq_mobv3_gn_safe_selector/safe_decision_scores | SafeDecisionScores | observed_snr, exploration_logits | [(None, 4), (None, 100)] | (None, 100) | 0 | {"dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null},"observed_indices":[0,33,66,99],"threshold":1.1885398626327515,"total_ports":100} |

All 32 serialized layer/container rows are listed above. The authoritative unique parameter total is 11,276.

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
| Built-in TFLite operators | 174 |

The TFLite converter lowers, decomposes, and fuses Keras layers into operators. Therefore, the exact human-readable layer architecture is the serialized Keras graph above; the TFLite operator count is not a second layer count.
