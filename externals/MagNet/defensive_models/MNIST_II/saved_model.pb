??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8ć
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?$
value?$B?$ B?$
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
iter

beta_1

beta_2
	 decay
!learning_ratemAmBmCmDmEmFvGvHvIvJvKvL
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
"metrics
regularization_losses
#non_trainable_variables
$layer_regularization_losses
trainable_variables

%layers
	variables
&layer_metrics
 
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
'metrics
regularization_losses
(non_trainable_variables
)layer_regularization_losses
trainable_variables

*layers
	variables
+layer_metrics
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
,metrics
regularization_losses
-non_trainable_variables
.layer_regularization_losses
trainable_variables

/layers
	variables
0layer_metrics
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
1metrics
regularization_losses
2non_trainable_variables
3layer_regularization_losses
trainable_variables

4layers
	variables
5layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

60
71
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	8total
	9count
:	variables
;	keras_api
D
	<total
	=count
>
_fn_kwargs
?	variables
@	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

80
91

:	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

?	variables
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_101084
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_101449
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_101540??
?r
?
"__inference__traced_restore_101540
file_prefix$
 assignvariableop_conv2d_5_kernel$
 assignvariableop_1_conv2d_5_bias&
"assignvariableop_2_conv2d_6_kernel$
 assignvariableop_3_conv2d_6_bias&
"assignvariableop_4_conv2d_7_kernel$
 assignvariableop_5_conv2d_7_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1.
*assignvariableop_15_adam_conv2d_5_kernel_m,
(assignvariableop_16_adam_conv2d_5_bias_m.
*assignvariableop_17_adam_conv2d_6_kernel_m,
(assignvariableop_18_adam_conv2d_6_bias_m.
*assignvariableop_19_adam_conv2d_7_kernel_m,
(assignvariableop_20_adam_conv2d_7_bias_m.
*assignvariableop_21_adam_conv2d_5_kernel_v,
(assignvariableop_22_adam_conv2d_5_bias_v.
*assignvariableop_23_adam_conv2d_6_kernel_v,
(assignvariableop_24_adam_conv2d_6_bias_v.
*assignvariableop_25_adam_conv2d_7_kernel_v,
(assignvariableop_26_adam_conv2d_7_bias_v
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_5_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_5_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_7_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_7_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_5_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_5_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_6_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_6_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_7_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_7_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*?
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
J
0__inference_conv2d_7_activity_regularizer_100729
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_101314

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_1007162
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_6_layer_call_fn_101303

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
0__inference_conv2d_5_activity_regularizer_100703
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
~
)__inference_conv2d_7_layer_call_fn_101334

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_100838

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?_
?
C__inference_model_1_layer_call_and_return_conditional_losses_101148

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3??conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_5/BiasAdd?
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_5/Sigmoid?
#conv2d_5/ActivityRegularizer/SquareSquareconv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_5/ActivityRegularizer/Square?
"conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_5/ActivityRegularizer/Const?
 conv2d_5/ActivityRegularizer/SumSum'conv2d_5/ActivityRegularizer/Square:y:0+conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_5/ActivityRegularizer/Sum?
"conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_5/ActivityRegularizer/mul/x?
 conv2d_5/ActivityRegularizer/mulMul+conv2d_5/ActivityRegularizer/mul/x:output:0)conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_5/ActivityRegularizer/mul?
"conv2d_5/ActivityRegularizer/ShapeShapeconv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv$conv2d_5/ActivityRegularizer/mul:z:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconv2d_5/Sigmoid:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_6/Sigmoid?
#conv2d_6/ActivityRegularizer/SquareSquareconv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_6/ActivityRegularizer/Square?
"conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_6/ActivityRegularizer/Const?
 conv2d_6/ActivityRegularizer/SumSum'conv2d_6/ActivityRegularizer/Square:y:0+conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_6/ActivityRegularizer/Sum?
"conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_6/ActivityRegularizer/mul/x?
 conv2d_6/ActivityRegularizer/mulMul+conv2d_6/ActivityRegularizer/mul/x:output:0)conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_6/ActivityRegularizer/mul?
"conv2d_6/ActivityRegularizer/ShapeShapeconv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv$conv2d_6/ActivityRegularizer/mul:z:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Sigmoid:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_7/BiasAdd?
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_7/Sigmoid?
#conv2d_7/ActivityRegularizer/SquareSquareconv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_7/ActivityRegularizer/Square?
"conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_7/ActivityRegularizer/Const?
 conv2d_7/ActivityRegularizer/SumSum'conv2d_7/ActivityRegularizer/Square:y:0+conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_7/ActivityRegularizer/Sum?
"conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_7/ActivityRegularizer/mul/x?
 conv2d_7/ActivityRegularizer/mulMul+conv2d_7/ActivityRegularizer/mul/x:output:0)conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_7/ActivityRegularizer/mul?
"conv2d_7/ActivityRegularizer/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv$conv2d_7/ActivityRegularizer/mul:z:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentityconv2d_7/Sigmoid:y:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_5_layer_call_fn_101272

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_101084
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1006902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_101263

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?_
?
C__inference_model_1_layer_call_and_return_conditional_losses_101212

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3??conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_5/BiasAdd?
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_5/Sigmoid?
#conv2d_5/ActivityRegularizer/SquareSquareconv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_5/ActivityRegularizer/Square?
"conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_5/ActivityRegularizer/Const?
 conv2d_5/ActivityRegularizer/SumSum'conv2d_5/ActivityRegularizer/Square:y:0+conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_5/ActivityRegularizer/Sum?
"conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_5/ActivityRegularizer/mul/x?
 conv2d_5/ActivityRegularizer/mulMul+conv2d_5/ActivityRegularizer/mul/x:output:0)conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_5/ActivityRegularizer/mul?
"conv2d_5/ActivityRegularizer/ShapeShapeconv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv$conv2d_5/ActivityRegularizer/mul:z:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconv2d_5/Sigmoid:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_6/Sigmoid?
#conv2d_6/ActivityRegularizer/SquareSquareconv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_6/ActivityRegularizer/Square?
"conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_6/ActivityRegularizer/Const?
 conv2d_6/ActivityRegularizer/SumSum'conv2d_6/ActivityRegularizer/Square:y:0+conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_6/ActivityRegularizer/Sum?
"conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_6/ActivityRegularizer/mul/x?
 conv2d_6/ActivityRegularizer/mulMul+conv2d_6/ActivityRegularizer/mul/x:output:0)conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_6/ActivityRegularizer/mul?
"conv2d_6/ActivityRegularizer/ShapeShapeconv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv$conv2d_6/ActivityRegularizer/mul:z:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Sigmoid:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_7/BiasAdd?
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_7/Sigmoid?
#conv2d_7/ActivityRegularizer/SquareSquareconv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_7/ActivityRegularizer/Square?
"conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_7/ActivityRegularizer/Const?
 conv2d_7/ActivityRegularizer/SumSum'conv2d_7/ActivityRegularizer/Square:y:0+conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_7/ActivityRegularizer/Sum?
"conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_7/ActivityRegularizer/mul/x?
 conv2d_7/ActivityRegularizer/mulMul+conv2d_7/ActivityRegularizer/mul/x:output:0)conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_7/ActivityRegularizer/mul?
"conv2d_7/ActivityRegularizer/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv$conv2d_7/ActivityRegularizer/mul:z:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentityconv2d_7/Sigmoid:y:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
C__inference_model_1_layer_call_and_return_conditional_losses_100973

inputs
conv2d_5_100930
conv2d_5_100932
conv2d_6_100943
conv2d_6_100945
conv2d_7_100956
conv2d_7_100958
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_100930conv2d_5_100932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442"
 conv2d_5/StatefulPartitionedCall?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_1007032.
,conv2d_5/ActivityRegularizer/PartitionedCall?
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_100943conv2d_6_100945*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912"
 conv2d_6/StatefulPartitionedCall?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_1007162.
,conv2d_6/ActivityRegularizer/PartitionedCall?
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_100956conv2d_7_100958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382"
 conv2d_7/StatefulPartitionedCall?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_1007292.
,conv2d_7/ActivityRegularizer/PartitionedCall?
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
0__inference_conv2d_6_activity_regularizer_100716
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_101345

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_1007292
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
C__inference_model_1_layer_call_and_return_conditional_losses_100878
input_2
conv2d_5_100767
conv2d_5_100769
conv2d_6_100814
conv2d_6_100816
conv2d_7_100861
conv2d_7_100863
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_5_100767conv2d_5_100769*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442"
 conv2d_5/StatefulPartitionedCall?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_1007032.
,conv2d_5/ActivityRegularizer/PartitionedCall?
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_100814conv2d_6_100816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912"
 conv2d_6/StatefulPartitionedCall?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_1007162.
,conv2d_6/ActivityRegularizer/PartitionedCall?
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_100861conv2d_7_100863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382"
 conv2d_7/StatefulPartitionedCall?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_1007292.
,conv2d_7/ActivityRegularizer/PartitionedCall?
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_101294

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_101057
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1010392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_100744

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_101325

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_101252

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1010392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
C__inference_model_1_layer_call_and_return_conditional_losses_101039

inputs
conv2d_5_100996
conv2d_5_100998
conv2d_6_101009
conv2d_6_101011
conv2d_7_101022
conv2d_7_101024
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_100996conv2d_5_100998*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442"
 conv2d_5/StatefulPartitionedCall?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_1007032.
,conv2d_5/ActivityRegularizer/PartitionedCall?
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_101009conv2d_6_101011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912"
 conv2d_6/StatefulPartitionedCall?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_1007162.
,conv2d_6/ActivityRegularizer/PartitionedCall?
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_101022conv2d_7_101024*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382"
 conv2d_7/StatefulPartitionedCall?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_1007292.
,conv2d_7/ActivityRegularizer/PartitionedCall?
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_100991
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1009732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_101283

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_1007032
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?b
?
!__inference__wrapped_model_100690
input_23
/model_1_conv2d_5_conv2d_readvariableop_resource4
0model_1_conv2d_5_biasadd_readvariableop_resource3
/model_1_conv2d_6_conv2d_readvariableop_resource4
0model_1_conv2d_6_biasadd_readvariableop_resource3
/model_1_conv2d_7_conv2d_readvariableop_resource4
0model_1_conv2d_7_biasadd_readvariableop_resource
identity??'model_1/conv2d_5/BiasAdd/ReadVariableOp?&model_1/conv2d_5/Conv2D/ReadVariableOp?'model_1/conv2d_6/BiasAdd/ReadVariableOp?&model_1/conv2d_6/Conv2D/ReadVariableOp?'model_1/conv2d_7/BiasAdd/ReadVariableOp?&model_1/conv2d_7/Conv2D/ReadVariableOp?
&model_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/conv2d_5/Conv2D/ReadVariableOp?
model_1/conv2d_5/Conv2DConv2Dinput_2.model_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_1/conv2d_5/Conv2D?
'model_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_5/BiasAdd/ReadVariableOp?
model_1/conv2d_5/BiasAddBiasAdd model_1/conv2d_5/Conv2D:output:0/model_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_5/BiasAdd?
model_1/conv2d_5/SigmoidSigmoid!model_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_5/Sigmoid?
+model_1/conv2d_5/ActivityRegularizer/SquareSquaremodel_1/conv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2-
+model_1/conv2d_5/ActivityRegularizer/Square?
*model_1/conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/conv2d_5/ActivityRegularizer/Const?
(model_1/conv2d_5/ActivityRegularizer/SumSum/model_1/conv2d_5/ActivityRegularizer/Square:y:03model_1/conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_5/ActivityRegularizer/Sum?
*model_1/conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02,
*model_1/conv2d_5/ActivityRegularizer/mul/x?
(model_1/conv2d_5/ActivityRegularizer/mulMul3model_1/conv2d_5/ActivityRegularizer/mul/x:output:01model_1/conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_5/ActivityRegularizer/mul?
*model_1/conv2d_5/ActivityRegularizer/ShapeShapemodel_1/conv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:2,
*model_1/conv2d_5/ActivityRegularizer/Shape?
8model_1/conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_1/conv2d_5/ActivityRegularizer/strided_slice/stack?
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_1?
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_2?
2model_1/conv2d_5/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_5/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_1/conv2d_5/ActivityRegularizer/strided_slice?
)model_1/conv2d_5/ActivityRegularizer/CastCast;model_1/conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_1/conv2d_5/ActivityRegularizer/Cast?
,model_1/conv2d_5/ActivityRegularizer/truedivRealDiv,model_1/conv2d_5/ActivityRegularizer/mul:z:0-model_1/conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_1/conv2d_5/ActivityRegularizer/truediv?
&model_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/conv2d_6/Conv2D/ReadVariableOp?
model_1/conv2d_6/Conv2DConv2Dmodel_1/conv2d_5/Sigmoid:y:0.model_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_1/conv2d_6/Conv2D?
'model_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_6/BiasAdd/ReadVariableOp?
model_1/conv2d_6/BiasAddBiasAdd model_1/conv2d_6/Conv2D:output:0/model_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_6/BiasAdd?
model_1/conv2d_6/SigmoidSigmoid!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_6/Sigmoid?
+model_1/conv2d_6/ActivityRegularizer/SquareSquaremodel_1/conv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2-
+model_1/conv2d_6/ActivityRegularizer/Square?
*model_1/conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/conv2d_6/ActivityRegularizer/Const?
(model_1/conv2d_6/ActivityRegularizer/SumSum/model_1/conv2d_6/ActivityRegularizer/Square:y:03model_1/conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_6/ActivityRegularizer/Sum?
*model_1/conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02,
*model_1/conv2d_6/ActivityRegularizer/mul/x?
(model_1/conv2d_6/ActivityRegularizer/mulMul3model_1/conv2d_6/ActivityRegularizer/mul/x:output:01model_1/conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_6/ActivityRegularizer/mul?
*model_1/conv2d_6/ActivityRegularizer/ShapeShapemodel_1/conv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:2,
*model_1/conv2d_6/ActivityRegularizer/Shape?
8model_1/conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_1/conv2d_6/ActivityRegularizer/strided_slice/stack?
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_1?
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_2?
2model_1/conv2d_6/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_6/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_1/conv2d_6/ActivityRegularizer/strided_slice?
)model_1/conv2d_6/ActivityRegularizer/CastCast;model_1/conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_1/conv2d_6/ActivityRegularizer/Cast?
,model_1/conv2d_6/ActivityRegularizer/truedivRealDiv,model_1/conv2d_6/ActivityRegularizer/mul:z:0-model_1/conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_1/conv2d_6/ActivityRegularizer/truediv?
&model_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/conv2d_7/Conv2D/ReadVariableOp?
model_1/conv2d_7/Conv2DConv2Dmodel_1/conv2d_6/Sigmoid:y:0.model_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_1/conv2d_7/Conv2D?
'model_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_7/BiasAdd/ReadVariableOp?
model_1/conv2d_7/BiasAddBiasAdd model_1/conv2d_7/Conv2D:output:0/model_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_7/BiasAdd?
model_1/conv2d_7/SigmoidSigmoid!model_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_7/Sigmoid?
+model_1/conv2d_7/ActivityRegularizer/SquareSquaremodel_1/conv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2-
+model_1/conv2d_7/ActivityRegularizer/Square?
*model_1/conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*model_1/conv2d_7/ActivityRegularizer/Const?
(model_1/conv2d_7/ActivityRegularizer/SumSum/model_1/conv2d_7/ActivityRegularizer/Square:y:03model_1/conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_7/ActivityRegularizer/Sum?
*model_1/conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02,
*model_1/conv2d_7/ActivityRegularizer/mul/x?
(model_1/conv2d_7/ActivityRegularizer/mulMul3model_1/conv2d_7/ActivityRegularizer/mul/x:output:01model_1/conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(model_1/conv2d_7/ActivityRegularizer/mul?
*model_1/conv2d_7/ActivityRegularizer/ShapeShapemodel_1/conv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:2,
*model_1/conv2d_7/ActivityRegularizer/Shape?
8model_1/conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_1/conv2d_7/ActivityRegularizer/strided_slice/stack?
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_1?
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_2?
2model_1/conv2d_7/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_7/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model_1/conv2d_7/ActivityRegularizer/strided_slice?
)model_1/conv2d_7/ActivityRegularizer/CastCast;model_1/conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)model_1/conv2d_7/ActivityRegularizer/Cast?
,model_1/conv2d_7/ActivityRegularizer/truedivRealDiv,model_1/conv2d_7/ActivityRegularizer/mul:z:0-model_1/conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2.
,model_1/conv2d_7/ActivityRegularizer/truediv?
IdentityIdentitymodel_1/conv2d_7/Sigmoid:y:0(^model_1/conv2d_5/BiasAdd/ReadVariableOp'^model_1/conv2d_5/Conv2D/ReadVariableOp(^model_1/conv2d_6/BiasAdd/ReadVariableOp'^model_1/conv2d_6/Conv2D/ReadVariableOp(^model_1/conv2d_7/BiasAdd/ReadVariableOp'^model_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2R
'model_1/conv2d_5/BiasAdd/ReadVariableOp'model_1/conv2d_5/BiasAdd/ReadVariableOp2P
&model_1/conv2d_5/Conv2D/ReadVariableOp&model_1/conv2d_5/Conv2D/ReadVariableOp2R
'model_1/conv2d_6/BiasAdd/ReadVariableOp'model_1/conv2d_6/BiasAdd/ReadVariableOp2P
&model_1/conv2d_6/Conv2D/ReadVariableOp&model_1/conv2d_6/Conv2D/ReadVariableOp2R
'model_1/conv2d_7/BiasAdd/ReadVariableOp'model_1/conv2d_7/BiasAdd/ReadVariableOp2P
&model_1/conv2d_7/Conv2D/ReadVariableOp&model_1/conv2d_7/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_100791

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
__inference__traced_save_101449
file_prefix.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : : : : ::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?D
?
C__inference_model_1_layer_call_and_return_conditional_losses_100924
input_2
conv2d_5_100881
conv2d_5_100883
conv2d_6_100894
conv2d_6_100896
conv2d_7_100907
conv2d_7_100909
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_5_100881conv2d_5_100883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1007442"
 conv2d_5/StatefulPartitionedCall?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_1007032.
,conv2d_5/ActivityRegularizer/PartitionedCall?
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_5/ActivityRegularizer/Shape?
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_5/ActivityRegularizer/strided_slice/stack?
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_1?
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_5/ActivityRegularizer/strided_slice/stack_2?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_5/ActivityRegularizer/strided_slice?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_5/ActivityRegularizer/Cast?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_5/ActivityRegularizer/truediv?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_100894conv2d_6_100896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1007912"
 conv2d_6/StatefulPartitionedCall?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_1007162.
,conv2d_6/ActivityRegularizer/PartitionedCall?
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_6/ActivityRegularizer/Shape?
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_6/ActivityRegularizer/strided_slice/stack?
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_1?
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_6/ActivityRegularizer/strided_slice/stack_2?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_6/ActivityRegularizer/strided_slice?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_6/ActivityRegularizer/Cast?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_6/ActivityRegularizer/truediv?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_100907conv2d_7_100909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1008382"
 conv2d_7/StatefulPartitionedCall?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_1007292.
,conv2d_7/ActivityRegularizer/PartitionedCall?
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_7/ActivityRegularizer/Shape?
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_7/ActivityRegularizer/strided_slice/stack?
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_1?
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_7/ActivityRegularizer/strided_slice/stack_2?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_7/ActivityRegularizer/strided_slice?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_7/ActivityRegularizer/Cast?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_7/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
(__inference_model_1_layer_call_fn_101232

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1009732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_28
serving_default_input_2:0?????????D
conv2d_78
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?5
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
M_default_save_signature
N__call__
*O&call_and_return_all_conditional_losses"?3
_tf_keras_network?2{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_7", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_7", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}
?
iter

beta_1

beta_2
	 decay
!learning_ratemAmBmCmDmEmFvGvHvIvJvKvL"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
"metrics
regularization_losses
#non_trainable_variables
$layer_regularization_losses
trainable_variables

%layers
	variables
&layer_metrics
N__call__
M_default_save_signature
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
,
Vserving_default"
signature_map
):'2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
'metrics
regularization_losses
(non_trainable_variables
)layer_regularization_losses
trainable_variables

*layers
	variables
+layer_metrics
P__call__
Wactivity_regularizer_fn
*Q&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_6/kernel
:2conv2d_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
,metrics
regularization_losses
-non_trainable_variables
.layer_regularization_losses
trainable_variables

/layers
	variables
0layer_metrics
R__call__
Yactivity_regularizer_fn
*S&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_7/kernel
:2conv2d_7/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
1metrics
regularization_losses
2non_trainable_variables
3layer_regularization_losses
trainable_variables

4layers
	variables
5layer_metrics
T__call__
[activity_regularizer_fn
*U&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	8total
	9count
:	variables
;	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	<total
	=count
>
_fn_kwargs
?	variables
@	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
80
91"
trackable_list_wrapper
-
:	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
-
?	variables"
_generic_user_object
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
.:,2Adam/conv2d_7/kernel/m
 :2Adam/conv2d_7/bias/m
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
.:,2Adam/conv2d_7/kernel/v
 :2Adam/conv2d_7/bias/v
?2?
!__inference__wrapped_model_100690?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_2?????????
?2?
(__inference_model_1_layer_call_fn_101057
(__inference_model_1_layer_call_fn_101232
(__inference_model_1_layer_call_fn_101252
(__inference_model_1_layer_call_fn_100991?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_101148
C__inference_model_1_layer_call_and_return_conditional_losses_100924
C__inference_model_1_layer_call_and_return_conditional_losses_101212
C__inference_model_1_layer_call_and_return_conditional_losses_100878?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_5_layer_call_fn_101272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_101283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_6_layer_call_fn_101303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_101314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_7_layer_call_fn_101334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_101345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_101084input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_5_activity_regularizer_100703?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_101263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_6_activity_regularizer_100716?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_101294?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_7_activity_regularizer_100729?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_101325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_1006908?5
.?+
)?&
input_2?????????
? ";?8
6
conv2d_7*?'
conv2d_7?????????]
0__inference_conv2d_5_activity_regularizer_100703)?
?
?
self
? "? ?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_101283z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_101263l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_5_layer_call_fn_101272_7?4
-?*
(?%
inputs?????????
? " ??????????]
0__inference_conv2d_6_activity_regularizer_100716)?
?
?
self
? "? ?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_101314z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_101294l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_6_layer_call_fn_101303_7?4
-?*
(?%
inputs?????????
? " ??????????]
0__inference_conv2d_7_activity_regularizer_100729)?
?
?
self
? "? ?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_101345z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_101325l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_7_layer_call_fn_101334_7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_model_1_layer_call_and_return_conditional_losses_100878?@?=
6?3
)?&
input_2?????????
p

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_100924?@?=
6?3
)?&
input_2?????????
p 

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_101148???<
5?2
(?%
inputs?????????
p

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_101212???<
5?2
(?%
inputs?????????
p 

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
(__inference_model_1_layer_call_fn_100991l@?=
6?3
)?&
input_2?????????
p

 
? " ???????????
(__inference_model_1_layer_call_fn_101057l@?=
6?3
)?&
input_2?????????
p 

 
? " ???????????
(__inference_model_1_layer_call_fn_101232k??<
5?2
(?%
inputs?????????
p

 
? " ???????????
(__inference_model_1_layer_call_fn_101252k??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
$__inference_signature_wrapper_101084?C?@
? 
9?6
4
input_2)?&
input_2?????????";?8
6
conv2d_7*?'
conv2d_7?????????