
import os
import setting
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from transformers.models.bert.modeling_tf_bert import TFBertEmbeddings,BertConfig
class PatchEmbedding(Layer):
    def __init__(self, image_size, patch_size, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embed = Conv2D(self.embed_dim, patch_size, patch_size)

        # 添加分类的token,会concat到image_tokens中,使得shape为[b,196+1,768]
        self.cls_token = self.add_weight('cls_token', shape=[1, 1, self.embed_dim],
                                         dtype='float32', initializer='random_normal',
                                         trainable=True)
        # pos_embedding与(image_tokens+cls_token)相加,所以shape也必须为[b,197,768]
        self.pos_embeding = self.add_weight('pos_embedding', shape=[1, self.n_patches + 1, self.embed_dim],
                                            dtype='float32', initializer='random_normal',
                                            trainable=True)

    def call(self, inputs):
        # patch_size=16, embed_dim=768
        # [b,224,224,3] -> [b,14,14,768]
        x = self.patch_embed(inputs)
        # [b,14,14,768] -> [b,196,768]
        b, h, w, _ = x.shape
        x = tf.reshape(x, shape=[b, h * w, self.embed_dim])
        # 1,1,768 -> b,1,768
        cls_tokens = tf.broadcast_to(self.cls_token, (b, 1, self.embed_dim))
        # -> b, 197, 768
        x = tf.concat([x, cls_tokens], axis=1)

        # 加上pos_embedding -> b, 197, 728
        x = x + self.pos_embeding

        return x

    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({"embed_dim": self.embed_dim,
                       "num_patches": self.n_patches,
                       })
        return config


# msa层的实现
class multiHead_self_attention(Layer):
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, **kwargs):
        super(multiHead_self_attention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.scale = self.head_dim ** (-0.5)  # q*k之后的变换系数

        self.qkv = Dense(self.all_head_dim * 3)
        self.proj = Dense(self.all_head_dim)

        self.attention_dropout = Dropout(attention_dropout)

        self.softmax = Softmax()

    def call(self, inputs):
        # -> b,197,768*3
        qkv = self.qkv(inputs)
        # q,k,v: b,197,768
        q, k, v = tf.split(qkv, 3, axis=-1)

        b, n_patches, all_head_dim = q.shape
        # q,k,v: b,197,768 -> b,197,num_heads, head_dim 假设num_heads=12
        # b,197,768 -> b,197,12,64
        q = tf.reshape(q, shape=[b, n_patches, self.num_heads, self.head_dim])
        k = tf.reshape(k, shape=[b, n_patches, self.num_heads, self.head_dim])
        v = tf.reshape(v, shape=[b, n_patches, self.num_heads, self.head_dim])

        # b,197,12,64 -> b,12,197,64
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        # -> b,12,12,64
        attention = tf.matmul(q, k, transpose_b=True)
        attention = self.scale * attention
        attention = self.softmax(attention)
        attention = self.attention_dropout(attention)
        # -> b,12,197,64
        out = tf.matmul(attention, v)
        # b,12,197,64 -> b,197,12,64
        out = tf.transpose(out, [0, 2, 1, 3])
        # b,197,12,64 -> b,197,768
        out = tf.reshape(out, shape=[b, n_patches, all_head_dim])

        out = self.proj(out)
        return out

    def get_config(self):
        config = super(multiHead_self_attention, self).get_config()
        config.update({"num_heads": self.num_heads,
                       "head_dim": self.head_dim,
                       "all_head_dim": self.all_head_dim,
                       "scale": self.scale
                       })
        return config


class MLP(Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

    def call(self, inputs):
        # 1,197,768 -> 1,197,768*4
        x = Dense(int(self.embed_dim * self.mlp_ratio))(inputs)
        x = gelu(x)
        x = Dropout(self.dropout)(x)

        # 1,197,768*4 - 1,197,768
        x = Dense(self.embed_dim)(x)
        x = Dropout(self.dropout)(x)

        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout
        })
class Visiontransformer(Layer):
    def __init__(self,num_heads,patch_size,embed_dim,layer_length,num_classes,isencoder=True):
        super(Visiontransformer, self).__init__()
        self.layer_length=layer_length
        self.patchembedding=PatchEmbedding(setting.img_h,patch_size=patch_size,embed_dim=embed_dim,name="patchAndPos_embedding")
        self.transformers = tf.keras.Sequential()
        self.norms_1=[LayerNormalization(name=f"layernorm{i}_1") for i in range(layer_length)]
        self.attentions=[multiHead_self_attention(embed_dim,num_heads,0,name=f"MSA{i}") for i in range(layer_length)]
        self.norms_2 = [LayerNormalization(name=f"layernorm{i}_2") for i in range(layer_length)]
        self.MLPS=[MLP(embed_dim=embed_dim,name=f"MLP{i}") for i in range(layer_length)]
        self.Dn1=Dense(num_classes,name="classifier",activation="softmax")
        self.isencoder=isencoder
    def call(self, inputs):
        x=self.patchembedding(inputs)
        for i in range(self.layer_length):
            x=self.norms_1[i](x)
            x1=self.attentions[i](x)
            x2=tf.concat([x,x1],axis=-1)
            x=self.norms_2[i](x)
            x=self.MLPS[i](x)
            x=tf.concat([x,x2],axis=-1)
        cls_token=x[:,0]
        if self.isencoder != True:
            # 1,768 -> 1, num_classes
            out = self.Dn1(cls_token)

        else:
            out = cls_token
        return out