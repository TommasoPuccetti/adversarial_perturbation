??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
 ?"serve*2.4.12unknown8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
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
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemmmnmompmq mr)ms*mt/mu0mvvwvxvyvzv{ v|)v}*v~/v0v?
 
F
0
1
2
3
4
 5
)6
*7
/8
09
F
0
1
2
3
4
 5
)6
*7
/8
09
?
:metrics

regularization_losses
;non_trainable_variables
<layer_regularization_losses
trainable_variables

=layers
	variables
>layer_metrics
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
?metrics
regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables

Blayers
	variables
Clayer_metrics
 
 
 
?
Dmetrics
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
trainable_variables

Glayers
	variables
Hlayer_metrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Imetrics
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables

Llayers
	variables
Mlayer_metrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
Nmetrics
!regularization_losses
Onon_trainable_variables
Player_regularization_losses
"trainable_variables

Qlayers
#	variables
Rlayer_metrics
 
 
 
?
Smetrics
%regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
&trainable_variables

Vlayers
'	variables
Wlayer_metrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
Xmetrics
+regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
,trainable_variables

[layers
-	variables
\layer_metrics
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
]metrics
1regularization_losses
^non_trainable_variables
_layer_regularization_losses
2trainable_variables

`layers
3	variables
alayer_metrics
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
b0
c1
 
 
8
0
1
2
3
4
5
6
7
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
 
 
 
 
4
	dtotal
	ecount
f	variables
g	keras_api
D
	htotal
	icount
j
_fn_kwargs
k	variables
l	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

f	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

h0
i1

k	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_50465
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_51046
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v*3
Tin,
*2(*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_51173??
??
?
!__inference__traced_restore_51173
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias&
"assignvariableop_8_conv2d_4_kernel$
 assignvariableop_9_conv2d_4_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1,
(assignvariableop_19_adam_conv2d_kernel_m*
&assignvariableop_20_adam_conv2d_bias_m.
*assignvariableop_21_adam_conv2d_1_kernel_m,
(assignvariableop_22_adam_conv2d_1_bias_m.
*assignvariableop_23_adam_conv2d_2_kernel_m,
(assignvariableop_24_adam_conv2d_2_bias_m.
*assignvariableop_25_adam_conv2d_3_kernel_m,
(assignvariableop_26_adam_conv2d_3_bias_m.
*assignvariableop_27_adam_conv2d_4_kernel_m,
(assignvariableop_28_adam_conv2d_4_bias_m,
(assignvariableop_29_adam_conv2d_kernel_v*
&assignvariableop_30_adam_conv2d_bias_v.
*assignvariableop_31_adam_conv2d_1_kernel_v,
(assignvariableop_32_adam_conv2d_1_bias_v.
*assignvariableop_33_adam_conv2d_2_kernel_v,
(assignvariableop_34_adam_conv2d_2_bias_v.
*assignvariableop_35_adam_conv2d_3_kernel_v,
(assignvariableop_36_adam_conv2d_3_bias_v.
*assignvariableop_37_adam_conv2d_4_kernel_v,
(assignvariableop_38_adam_conv2d_4_bias_v
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39?
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_50886

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
T0*A
_output_shapes/
-:+???????????????????????????*
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
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_50875

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_498812
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_50430
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *K
_output_shapes9
7:+???????????????????????????: : : : : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_504022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
I
-__inference_up_sampling2d_layer_call_fn_49868

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_498622
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_49798
input_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource1
-model_conv2d_4_conv2d_readvariableop_resource2
.model_conv2d_4_biasadd_readvariableop_resource
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d/BiasAdd?
model/conv2d/SigmoidSigmoidmodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d/Sigmoid?
'model/conv2d/ActivityRegularizer/SquareSquaremodel/conv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2)
'model/conv2d/ActivityRegularizer/Square?
&model/conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/conv2d/ActivityRegularizer/Const?
$model/conv2d/ActivityRegularizer/SumSum+model/conv2d/ActivityRegularizer/Square:y:0/model/conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2&
$model/conv2d/ActivityRegularizer/Sum?
&model/conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02(
&model/conv2d/ActivityRegularizer/mul/x?
$model/conv2d/ActivityRegularizer/mulMul/model/conv2d/ActivityRegularizer/mul/x:output:0-model/conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$model/conv2d/ActivityRegularizer/mul?
&model/conv2d/ActivityRegularizer/ShapeShapemodel/conv2d/Sigmoid:y:0*
T0*
_output_shapes
:2(
&model/conv2d/ActivityRegularizer/Shape?
4model/conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model/conv2d/ActivityRegularizer/strided_slice/stack?
6model/conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6model/conv2d/ActivityRegularizer/strided_slice/stack_1?
6model/conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model/conv2d/ActivityRegularizer/strided_slice/stack_2?
.model/conv2d/ActivityRegularizer/strided_sliceStridedSlice/model/conv2d/ActivityRegularizer/Shape:output:0=model/conv2d/ActivityRegularizer/strided_slice/stack:output:0?model/conv2d/ActivityRegularizer/strided_slice/stack_1:output:0?model/conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.model/conv2d/ActivityRegularizer/strided_slice?
%model/conv2d/ActivityRegularizer/CastCast7model/conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%model/conv2d/ActivityRegularizer/Cast?
(model/conv2d/ActivityRegularizer/truedivRealDiv(model/conv2d/ActivityRegularizer/mul:z:0)model/conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2*
(model/conv2d/ActivityRegularizer/truediv?
model/average_pooling2d/AvgPoolAvgPoolmodel/conv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2!
model/average_pooling2d/AvgPool?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2D(model/average_pooling2d/AvgPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d_1/BiasAdd?
model/conv2d_1/SigmoidSigmoidmodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d_1/Sigmoid?
)model/conv2d_1/ActivityRegularizer/SquareSquaremodel/conv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2+
)model/conv2d_1/ActivityRegularizer/Square?
(model/conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/conv2d_1/ActivityRegularizer/Const?
&model/conv2d_1/ActivityRegularizer/SumSum-model/conv2d_1/ActivityRegularizer/Square:y:01model/conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_1/ActivityRegularizer/Sum?
(model/conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02*
(model/conv2d_1/ActivityRegularizer/mul/x?
&model/conv2d_1/ActivityRegularizer/mulMul1model/conv2d_1/ActivityRegularizer/mul/x:output:0/model/conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_1/ActivityRegularizer/mul?
(model/conv2d_1/ActivityRegularizer/ShapeShapemodel/conv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:2*
(model/conv2d_1/ActivityRegularizer/Shape?
6model/conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/conv2d_1/ActivityRegularizer/strided_slice/stack?
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_1?
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_2?
0model/conv2d_1/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_1/ActivityRegularizer/Shape:output:0?model/conv2d_1/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/conv2d_1/ActivityRegularizer/strided_slice?
'model/conv2d_1/ActivityRegularizer/CastCast9model/conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'model/conv2d_1/ActivityRegularizer/Cast?
*model/conv2d_1/ActivityRegularizer/truedivRealDiv*model/conv2d_1/ActivityRegularizer/mul:z:0+model/conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2,
*model/conv2d_1/ActivityRegularizer/truediv?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2Dmodel/conv2d_1/Sigmoid:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d_2/BiasAdd?
model/conv2d_2/SigmoidSigmoidmodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d_2/Sigmoid?
)model/conv2d_2/ActivityRegularizer/SquareSquaremodel/conv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2+
)model/conv2d_2/ActivityRegularizer/Square?
(model/conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/conv2d_2/ActivityRegularizer/Const?
&model/conv2d_2/ActivityRegularizer/SumSum-model/conv2d_2/ActivityRegularizer/Square:y:01model/conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_2/ActivityRegularizer/Sum?
(model/conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02*
(model/conv2d_2/ActivityRegularizer/mul/x?
&model/conv2d_2/ActivityRegularizer/mulMul1model/conv2d_2/ActivityRegularizer/mul/x:output:0/model/conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_2/ActivityRegularizer/mul?
(model/conv2d_2/ActivityRegularizer/ShapeShapemodel/conv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2*
(model/conv2d_2/ActivityRegularizer/Shape?
6model/conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/conv2d_2/ActivityRegularizer/strided_slice/stack?
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_1?
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_2?
0model/conv2d_2/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_2/ActivityRegularizer/Shape:output:0?model/conv2d_2/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/conv2d_2/ActivityRegularizer/strided_slice?
'model/conv2d_2/ActivityRegularizer/CastCast9model/conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'model/conv2d_2/ActivityRegularizer/Cast?
*model/conv2d_2/ActivityRegularizer/truedivRealDiv*model/conv2d_2/ActivityRegularizer/mul:z:0+model/conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2,
*model/conv2d_2/ActivityRegularizer/truediv?
model/up_sampling2d/ShapeShapemodel/conv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2
model/up_sampling2d/Shape?
'model/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'model/up_sampling2d/strided_slice/stack?
)model/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/up_sampling2d/strided_slice/stack_1?
)model/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/up_sampling2d/strided_slice/stack_2?
!model/up_sampling2d/strided_sliceStridedSlice"model/up_sampling2d/Shape:output:00model/up_sampling2d/strided_slice/stack:output:02model/up_sampling2d/strided_slice/stack_1:output:02model/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!model/up_sampling2d/strided_slice?
model/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model/up_sampling2d/Const?
model/up_sampling2d/mulMul*model/up_sampling2d/strided_slice:output:0"model/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model/up_sampling2d/mul?
0model/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbormodel/conv2d_2/Sigmoid:y:0model/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(22
0model/up_sampling2d/resize/ResizeNearestNeighbor?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOp?
model/conv2d_3/Conv2DConv2DAmodel/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model/conv2d_3/Conv2D?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d_3/BiasAdd?
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d_3/Sigmoid?
)model/conv2d_3/ActivityRegularizer/SquareSquaremodel/conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2+
)model/conv2d_3/ActivityRegularizer/Square?
(model/conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/conv2d_3/ActivityRegularizer/Const?
&model/conv2d_3/ActivityRegularizer/SumSum-model/conv2d_3/ActivityRegularizer/Square:y:01model/conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_3/ActivityRegularizer/Sum?
(model/conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02*
(model/conv2d_3/ActivityRegularizer/mul/x?
&model/conv2d_3/ActivityRegularizer/mulMul1model/conv2d_3/ActivityRegularizer/mul/x:output:0/model/conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_3/ActivityRegularizer/mul?
(model/conv2d_3/ActivityRegularizer/ShapeShapemodel/conv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:2*
(model/conv2d_3/ActivityRegularizer/Shape?
6model/conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/conv2d_3/ActivityRegularizer/strided_slice/stack?
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_1?
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_2?
0model/conv2d_3/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_3/ActivityRegularizer/Shape:output:0?model/conv2d_3/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/conv2d_3/ActivityRegularizer/strided_slice?
'model/conv2d_3/ActivityRegularizer/CastCast9model/conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'model/conv2d_3/ActivityRegularizer/Cast?
*model/conv2d_3/ActivityRegularizer/truedivRealDiv*model/conv2d_3/ActivityRegularizer/mul:z:0+model/conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2,
*model/conv2d_3/ActivityRegularizer/truediv?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOp?
model/conv2d_4/Conv2DConv2Dmodel/conv2d_3/Sigmoid:y:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model/conv2d_4/Conv2D?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d_4/BiasAdd?
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d_4/Sigmoid?
)model/conv2d_4/ActivityRegularizer/SquareSquaremodel/conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2+
)model/conv2d_4/ActivityRegularizer/Square?
(model/conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(model/conv2d_4/ActivityRegularizer/Const?
&model/conv2d_4/ActivityRegularizer/SumSum-model/conv2d_4/ActivityRegularizer/Square:y:01model/conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_4/ActivityRegularizer/Sum?
(model/conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02*
(model/conv2d_4/ActivityRegularizer/mul/x?
&model/conv2d_4/ActivityRegularizer/mulMul1model/conv2d_4/ActivityRegularizer/mul/x:output:0/model/conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&model/conv2d_4/ActivityRegularizer/mul?
(model/conv2d_4/ActivityRegularizer/ShapeShapemodel/conv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:2*
(model/conv2d_4/ActivityRegularizer/Shape?
6model/conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/conv2d_4/ActivityRegularizer/strided_slice/stack?
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_1?
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_2?
0model/conv2d_4/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_4/ActivityRegularizer/Shape:output:0?model/conv2d_4/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/conv2d_4/ActivityRegularizer/strided_slice?
'model/conv2d_4/ActivityRegularizer/CastCast9model/conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'model/conv2d_4/ActivityRegularizer/Cast?
*model/conv2d_4/ActivityRegularizer/truedivRealDiv*model/conv2d_4/ActivityRegularizer/mul:z:0+model/conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2,
*model/conv2d_4/ActivityRegularizer/truediv?
IdentityIdentitymodel/conv2d_4/Sigmoid:y:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
G
-__inference_conv2d_activity_regularizer_49811
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
A__inference_conv2d_layer_call_and_return_conditional_losses_50762

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
?
I
/__inference_conv2d_1_activity_regularizer_49836
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
??
?
@__inference_model_layer_call_and_return_conditional_losses_50578

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAdd~
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Sigmoid?
!conv2d/ActivityRegularizer/SquareSquareconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2#
!conv2d/ActivityRegularizer/Square?
 conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2d/ActivityRegularizer/Const?
conv2d/ActivityRegularizer/SumSum%conv2d/ActivityRegularizer/Square:y:0)conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d/ActivityRegularizer/Sum?
 conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02"
 conv2d/ActivityRegularizer/mul/x?
conv2d/ActivityRegularizer/mulMul)conv2d/ActivityRegularizer/mul/x:output:0'conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d/ActivityRegularizer/mul?
 conv2d/ActivityRegularizer/ShapeShapeconv2d/Sigmoid:y:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv"conv2d/ActivityRegularizer/mul:z:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
average_pooling2d/AvgPoolAvgPoolconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_1/BiasAdd?
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_1/Sigmoid?
#conv2d_1/ActivityRegularizer/SquareSquareconv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_1/ActivityRegularizer/Square?
"conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_1/ActivityRegularizer/Const?
 conv2d_1/ActivityRegularizer/SumSum'conv2d_1/ActivityRegularizer/Square:y:0+conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_1/ActivityRegularizer/Sum?
"conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_1/ActivityRegularizer/mul/x?
 conv2d_1/ActivityRegularizer/mulMul+conv2d_1/ActivityRegularizer/mul/x:output:0)conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_1/ActivityRegularizer/mul?
"conv2d_1/ActivityRegularizer/ShapeShapeconv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv$conv2d_1/ActivityRegularizer/mul:z:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Sigmoid:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd?
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Sigmoid?
#conv2d_2/ActivityRegularizer/SquareSquareconv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_2/ActivityRegularizer/Square?
"conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_2/ActivityRegularizer/Const?
 conv2d_2/ActivityRegularizer/SumSum'conv2d_2/ActivityRegularizer/Square:y:0+conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_2/ActivityRegularizer/Sum?
"conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_2/ActivityRegularizer/mul/x?
 conv2d_2/ActivityRegularizer/mulMul+conv2d_2/ActivityRegularizer/mul/x:output:0)conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_2/ActivityRegularizer/mul?
"conv2d_2/ActivityRegularizer/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv$conv2d_2/ActivityRegularizer/mul:z:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truedivn
up_sampling2d/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Sigmoid:y:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_3/BiasAdd?
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_3/Sigmoid?
#conv2d_3/ActivityRegularizer/SquareSquareconv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_3/ActivityRegularizer/Square?
"conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_3/ActivityRegularizer/Const?
 conv2d_3/ActivityRegularizer/SumSum'conv2d_3/ActivityRegularizer/Square:y:0+conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_3/ActivityRegularizer/Sum?
"conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_3/ActivityRegularizer/mul/x?
 conv2d_3/ActivityRegularizer/mulMul+conv2d_3/ActivityRegularizer/mul/x:output:0)conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_3/ActivityRegularizer/mul?
"conv2d_3/ActivityRegularizer/ShapeShapeconv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv$conv2d_3/ActivityRegularizer/mul:z:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconv2d_3/Sigmoid:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd?
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Sigmoid?
#conv2d_4/ActivityRegularizer/SquareSquareconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_4/ActivityRegularizer/Square?
"conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_4/ActivityRegularizer/Const?
 conv2d_4/ActivityRegularizer/SumSum'conv2d_4/ActivityRegularizer/Square:y:0+conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_4/ActivityRegularizer/Sum?
"conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_4/ActivityRegularizer/mul/x?
 conv2d_4/ActivityRegularizer/mulMul+conv2d_4/ActivityRegularizer/mul/x:output:0)conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_4/ActivityRegularizer/mul?
"conv2d_4/ActivityRegularizer/ShapeShapeconv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv$conv2d_4/ActivityRegularizer/mul:z:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentityconv2d_4/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_50813

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_498362
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_50004

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
:?????????*
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
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_49909

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
?
{
&__inference_conv2d_layer_call_fn_50771

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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092
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
}
(__inference_conv2d_4_layer_call_fn_50895

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_50906

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_498942
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_2_layer_call_fn_50833

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
/__inference_conv2d_4_activity_regularizer_49894
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
?S
?
__inference__traced_save_51046
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::: : : : : : : : : ::::::::::::::::::::: 2(
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::(

_output_shapes
: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_50691

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAdd~
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Sigmoid?
!conv2d/ActivityRegularizer/SquareSquareconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2#
!conv2d/ActivityRegularizer/Square?
 conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2d/ActivityRegularizer/Const?
conv2d/ActivityRegularizer/SumSum%conv2d/ActivityRegularizer/Square:y:0)conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d/ActivityRegularizer/Sum?
 conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02"
 conv2d/ActivityRegularizer/mul/x?
conv2d/ActivityRegularizer/mulMul)conv2d/ActivityRegularizer/mul/x:output:0'conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d/ActivityRegularizer/mul?
 conv2d/ActivityRegularizer/ShapeShapeconv2d/Sigmoid:y:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv"conv2d/ActivityRegularizer/mul:z:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
average_pooling2d/AvgPoolAvgPoolconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_1/BiasAdd?
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_1/Sigmoid?
#conv2d_1/ActivityRegularizer/SquareSquareconv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_1/ActivityRegularizer/Square?
"conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_1/ActivityRegularizer/Const?
 conv2d_1/ActivityRegularizer/SumSum'conv2d_1/ActivityRegularizer/Square:y:0+conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_1/ActivityRegularizer/Sum?
"conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_1/ActivityRegularizer/mul/x?
 conv2d_1/ActivityRegularizer/mulMul+conv2d_1/ActivityRegularizer/mul/x:output:0)conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_1/ActivityRegularizer/mul?
"conv2d_1/ActivityRegularizer/ShapeShapeconv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv$conv2d_1/ActivityRegularizer/mul:z:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Sigmoid:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd?
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Sigmoid?
#conv2d_2/ActivityRegularizer/SquareSquareconv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_2/ActivityRegularizer/Square?
"conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_2/ActivityRegularizer/Const?
 conv2d_2/ActivityRegularizer/SumSum'conv2d_2/ActivityRegularizer/Square:y:0+conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_2/ActivityRegularizer/Sum?
"conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_2/ActivityRegularizer/mul/x?
 conv2d_2/ActivityRegularizer/mulMul+conv2d_2/ActivityRegularizer/mul/x:output:0)conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_2/ActivityRegularizer/mul?
"conv2d_2/ActivityRegularizer/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv$conv2d_2/ActivityRegularizer/mul:z:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truedivn
up_sampling2d/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Sigmoid:y:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_3/BiasAdd?
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_3/Sigmoid?
#conv2d_3/ActivityRegularizer/SquareSquareconv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_3/ActivityRegularizer/Square?
"conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_3/ActivityRegularizer/Const?
 conv2d_3/ActivityRegularizer/SumSum'conv2d_3/ActivityRegularizer/Square:y:0+conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_3/ActivityRegularizer/Sum?
"conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_3/ActivityRegularizer/mul/x?
 conv2d_3/ActivityRegularizer/mulMul+conv2d_3/ActivityRegularizer/mul/x:output:0)conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_3/ActivityRegularizer/mul?
"conv2d_3/ActivityRegularizer/ShapeShapeconv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv$conv2d_3/ActivityRegularizer/mul:z:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconv2d_3/Sigmoid:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd?
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Sigmoid?
#conv2d_4/ActivityRegularizer/SquareSquareconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????2%
#conv2d_4/ActivityRegularizer/Square?
"conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_4/ActivityRegularizer/Const?
 conv2d_4/ActivityRegularizer/SumSum'conv2d_4/ActivityRegularizer/Square:y:0+conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_4/ActivityRegularizer/Sum?
"conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?02$
"conv2d_4/ActivityRegularizer/mul/x?
 conv2d_4/ActivityRegularizer/mulMul+conv2d_4/ActivityRegularizer/mul/x:output:0)conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_4/ActivityRegularizer/mul?
"conv2d_4/ActivityRegularizer/ShapeShapeconv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv$conv2d_4/ActivityRegularizer/mul:z:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentityconv2d_4/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_50052

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
T0*A
_output_shapes/
-:+???????????????????????????*
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
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_50099

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
T0*A
_output_shapes/
-:+???????????????????????????*
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
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_49862

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?y
?
@__inference_model_layer_call_and_return_conditional_losses_50296

inputs
conv2d_50223
conv2d_50225
conv2d_1_50237
conv2d_1_50239
conv2d_2_50250
conv2d_2_50252
conv2d_3_50264
conv2d_3_50266
conv2d_4_50277
conv2d_4_50279
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_50223conv2d_50225*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092 
conv2d/StatefulPartitionedCall?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_498112,
*conv2d/ActivityRegularizer/PartitionedCall?
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_498172#
!average_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_50237conv2d_1_50239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572"
 conv2d_1/StatefulPartitionedCall?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_498362.
,conv2d_1/ActivityRegularizer/PartitionedCall?
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_50250conv2d_2_50252*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042"
 conv2d_2/StatefulPartitionedCall?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_498492.
,conv2d_2/ActivityRegularizer/PartitionedCall?
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truediv?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_498622
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_50264conv2d_3_50266*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522"
 conv2d_3/StatefulPartitionedCall?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_498812.
,conv2d_3/ActivityRegularizer/PartitionedCall?
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_50277conv2d_4_50279*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992"
 conv2d_4/StatefulPartitionedCall?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_498942.
,conv2d_4/ActivityRegularizer/PartitionedCall?
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_50721

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *K
_output_shapes9
7:+???????????????????????????: : : : : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_502962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_49957

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
:?????????*
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
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_3_layer_call_fn_50864

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_50751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *K
_output_shapes9
7:+???????????????????????????: : : : : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_504022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_average_pooling2d_layer_call_fn_49823

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_498172
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_50844

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_498492
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?y
?
@__inference_model_layer_call_and_return_conditional_losses_50217
input_1
conv2d_50144
conv2d_50146
conv2d_1_50158
conv2d_1_50160
conv2d_2_50171
conv2d_2_50173
conv2d_3_50185
conv2d_3_50187
conv2d_4_50198
conv2d_4_50200
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_50144conv2d_50146*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092 
conv2d/StatefulPartitionedCall?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_498112,
*conv2d/ActivityRegularizer/PartitionedCall?
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_498172#
!average_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_50158conv2d_1_50160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572"
 conv2d_1/StatefulPartitionedCall?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_498362.
,conv2d_1/ActivityRegularizer/PartitionedCall?
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_50171conv2d_2_50173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042"
 conv2d_2/StatefulPartitionedCall?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_498492.
,conv2d_2/ActivityRegularizer/PartitionedCall?
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truediv?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_498622
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_50185conv2d_3_50187*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522"
 conv2d_3/StatefulPartitionedCall?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_498812.
,conv2d_3/ActivityRegularizer/PartitionedCall?
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_50198conv2d_4_50200*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992"
 conv2d_4/StatefulPartitionedCall?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_498942.
,conv2d_4/ActivityRegularizer/PartitionedCall?
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_49817

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
I
/__inference_conv2d_3_activity_regularizer_49881
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
?y
?
@__inference_model_layer_call_and_return_conditional_losses_50141
input_1
conv2d_49932
conv2d_49934
conv2d_1_49980
conv2d_1_49982
conv2d_2_50027
conv2d_2_50029
conv2d_3_50075
conv2d_3_50077
conv2d_4_50122
conv2d_4_50124
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_49932conv2d_49934*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092 
conv2d/StatefulPartitionedCall?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_498112,
*conv2d/ActivityRegularizer/PartitionedCall?
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_498172#
!average_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_49980conv2d_1_49982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572"
 conv2d_1/StatefulPartitionedCall?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_498362.
,conv2d_1/ActivityRegularizer/PartitionedCall?
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_50027conv2d_2_50029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042"
 conv2d_2/StatefulPartitionedCall?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_498492.
,conv2d_2/ActivityRegularizer/PartitionedCall?
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truediv?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_498622
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_50075conv2d_3_50077*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522"
 conv2d_3/StatefulPartitionedCall?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_498812.
,conv2d_3/ActivityRegularizer/PartitionedCall?
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_50122conv2d_4_50124*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992"
 conv2d_4/StatefulPartitionedCall?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_498942.
,conv2d_4/ActivityRegularizer/PartitionedCall?
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_50824

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
:?????????*
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
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_layer_call_and_return_all_conditional_losses_50782

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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092
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
GPU2*0J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_498112
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
?
}
(__inference_conv2d_1_layer_call_fn_50802

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
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_50855

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
T0*A
_output_shapes/
-:+???????????????????????????*
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
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?y
?
@__inference_model_layer_call_and_return_conditional_losses_50402

inputs
conv2d_50329
conv2d_50331
conv2d_1_50343
conv2d_1_50345
conv2d_2_50356
conv2d_2_50358
conv2d_3_50370
conv2d_3_50372
conv2d_4_50383
conv2d_4_50385
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_50329conv2d_50331*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_499092 
conv2d/StatefulPartitionedCall?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_498112,
*conv2d/ActivityRegularizer/PartitionedCall?
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2"
 conv2d/ActivityRegularizer/Shape?
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.conv2d/ActivityRegularizer/strided_slice/stack?
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_1?
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0conv2d/ActivityRegularizer/strided_slice/stack_2?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(conv2d/ActivityRegularizer/strided_slice?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d/ActivityRegularizer/Cast?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2$
"conv2d/ActivityRegularizer/truediv?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_498172#
!average_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_50343conv2d_1_50345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_499572"
 conv2d_1/StatefulPartitionedCall?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_498362.
,conv2d_1/ActivityRegularizer/PartitionedCall?
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_1/ActivityRegularizer/Shape?
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_1/ActivityRegularizer/strided_slice/stack?
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_1?
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_1/ActivityRegularizer/strided_slice/stack_2?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_1/ActivityRegularizer/strided_slice?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_1/ActivityRegularizer/Cast?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_1/ActivityRegularizer/truediv?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_50356conv2d_2_50358*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_500042"
 conv2d_2/StatefulPartitionedCall?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_498492.
,conv2d_2/ActivityRegularizer/PartitionedCall?
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_2/ActivityRegularizer/Shape?
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_2/ActivityRegularizer/strided_slice/stack?
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_1?
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_2/ActivityRegularizer/strided_slice/stack_2?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_2/ActivityRegularizer/strided_slice?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_2/ActivityRegularizer/Cast?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_2/ActivityRegularizer/truediv?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_498622
up_sampling2d/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_50370conv2d_3_50372*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_500522"
 conv2d_3/StatefulPartitionedCall?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_498812.
,conv2d_3/ActivityRegularizer/PartitionedCall?
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_3/ActivityRegularizer/Shape?
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_3/ActivityRegularizer/strided_slice/stack?
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_1?
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_3/ActivityRegularizer/strided_slice/stack_2?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_3/ActivityRegularizer/strided_slice?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_3/ActivityRegularizer/Cast?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_3/ActivityRegularizer/truediv?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_50383conv2d_4_50385*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_500992"
 conv2d_4/StatefulPartitionedCall?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_498942.
,conv2d_4/ActivityRegularizer/PartitionedCall?
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"conv2d_4/ActivityRegularizer/Shape?
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv2d_4/ActivityRegularizer/strided_slice/stack?
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_1?
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv2d_4/ActivityRegularizer/strided_slice/stack_2?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*conv2d_4/ActivityRegularizer/strided_slice?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!conv2d_4/ActivityRegularizer/Cast?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$conv2d_4/ActivityRegularizer/truediv?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_4?

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*V
_input_shapesE
C:?????????::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_50465
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_497982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
%__inference_model_layer_call_fn_50324
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *K
_output_shapes9
7:+???????????????????????????: : : : : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_502962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
I
/__inference_conv2d_2_activity_regularizer_49849
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_50793

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
:?????????*
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
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
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
input_18
serving_default_input_1:0?????????D
conv2d_48
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?Y
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?V
_tf_keras_network?V{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_4", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 3]}}
?

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 3]}}
?
%regularization_losses
&trainable_variables
'	variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}
?

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 9.999999717180685e-10}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemmmnmompmq mr)ms*mt/mu0mvvwvxvyvzv{ v|)v}*v~/v0v?"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
 5
)6
*7
/8
09"
trackable_list_wrapper
f
0
1
2
3
4
 5
)6
*7
/8
09"
trackable_list_wrapper
?
:metrics

regularization_losses
;non_trainable_variables
<layer_regularization_losses
trainable_variables

=layers
	variables
>layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?metrics
regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables

Blayers
	variables
Clayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dmetrics
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
trainable_variables

Glayers
	variables
Hlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Imetrics
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables

Llayers
	variables
Mlayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
Nmetrics
!regularization_losses
Onon_trainable_variables
Player_regularization_losses
"trainable_variables

Qlayers
#	variables
Rlayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Smetrics
%regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
&trainable_variables

Vlayers
'	variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
Xmetrics
+regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
,trainable_variables

[layers
-	variables
\layer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
]metrics
1regularization_losses
^non_trainable_variables
_layer_regularization_losses
2trainable_variables

`layers
3	variables
alayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
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
	dtotal
	ecount
f	variables
g	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	htotal
	icount
j
_fn_kwargs
k	variables
l	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
h0
i1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
?2?
 __inference__wrapped_model_49798?
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
input_1?????????
?2?
%__inference_model_layer_call_fn_50751
%__inference_model_layer_call_fn_50430
%__inference_model_layer_call_fn_50324
%__inference_model_layer_call_fn_50721?
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
@__inference_model_layer_call_and_return_conditional_losses_50578
@__inference_model_layer_call_and_return_conditional_losses_50217
@__inference_model_layer_call_and_return_conditional_losses_50691
@__inference_model_layer_call_and_return_conditional_losses_50141?
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
&__inference_conv2d_layer_call_fn_50771?
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
E__inference_conv2d_layer_call_and_return_all_conditional_losses_50782?
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
?2?
1__inference_average_pooling2d_layer_call_fn_49823?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_49817?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2d_1_layer_call_fn_50802?
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
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_50813?
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
(__inference_conv2d_2_layer_call_fn_50833?
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
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_50844?
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
?2?
-__inference_up_sampling2d_layer_call_fn_49868?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_49862?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2d_3_layer_call_fn_50864?
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
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_50875?
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
(__inference_conv2d_4_layer_call_fn_50895?
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
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_50906?
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
#__inference_signature_wrapper_50465input_1"?
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
-__inference_conv2d_activity_regularizer_49811?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_50762?
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
/__inference_conv2d_1_activity_regularizer_49836?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_50793?
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
/__inference_conv2d_2_activity_regularizer_49849?
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_50824?
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
/__inference_conv2d_3_activity_regularizer_49881?
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_50855?
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
/__inference_conv2d_4_activity_regularizer_49894?
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_50886?
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
 __inference__wrapped_model_49798?
 )*/08?5
.?+
)?&
input_1?????????
? ";?8
6
conv2d_4*?'
conv2d_4??????????
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_49817?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_average_pooling2d_layer_call_fn_49823?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????\
/__inference_conv2d_1_activity_regularizer_49836)?
?
?
self
? "? ?
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_50813z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_50793l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_1_layer_call_fn_50802_7?4
-?*
(?%
inputs?????????
? " ??????????\
/__inference_conv2d_2_activity_regularizer_49849)?
?
?
self
? "? ?
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_50844z 7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_50824l 7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_2_layer_call_fn_50833_ 7?4
-?*
(?%
inputs?????????
? " ??????????\
/__inference_conv2d_3_activity_regularizer_49881)?
?
?
self
? "? ?
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_50875?)*I?F
??<
:?7
inputs+???????????????????????????
? "M?J
5?2
0+???????????????????????????
?
?	
1/0 ?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_50855?)*I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_3_layer_call_fn_50864?)*I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????\
/__inference_conv2d_4_activity_regularizer_49894)?
?
?
self
? "? ?
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_50906?/0I?F
??<
:?7
inputs+???????????????????????????
? "M?J
5?2
0+???????????????????????????
?
?	
1/0 ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_50886?/0I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_4_layer_call_fn_50895?/0I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????Z
-__inference_conv2d_activity_regularizer_49811)?
?
?
self
? "? ?
E__inference_conv2d_layer_call_and_return_all_conditional_losses_50782z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
A__inference_conv2d_layer_call_and_return_conditional_losses_50762l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_50771_7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_model_layer_call_and_return_conditional_losses_50141?
 )*/0@?=
6?3
)?&
input_1?????????
p

 
? "???
5?2
0+???????????????????????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_50217?
 )*/0@?=
6?3
)?&
input_1?????????
p 

 
? "???
5?2
0+???????????????????????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_50578?
 )*/0??<
5?2
(?%
inputs?????????
p

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_50691?
 )*/0??<
5?2
(?%
inputs?????????
p 

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
%__inference_model_layer_call_fn_50324?
 )*/0@?=
6?3
)?&
input_1?????????
p

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_50430?
 )*/0@?=
6?3
)?&
input_1?????????
p 

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_50721?
 )*/0??<
5?2
(?%
inputs?????????
p

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_50751?
 )*/0??<
5?2
(?%
inputs?????????
p 

 
? "2?/+????????????????????????????
#__inference_signature_wrapper_50465?
 )*/0C?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
conv2d_4*?'
conv2d_4??????????
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_49862?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_up_sampling2d_layer_call_fn_49868?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????