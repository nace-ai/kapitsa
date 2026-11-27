import json,struct,pathlib
root=pathlib.Path(__file__).resolve().parent/'fixtures';root.mkdir(exist_ok=True)
cases={
 'valid':('U32',[2,8],[2,1],[2,1]),
 'scalar':('U32',[],[1,1],[1,1]),
 'rank1':('U32',[8],[1,1],[1,1]),
 'scale_rows':('U32',[2,8],[1,1],[2,1]),
 'bias_groups':('U32',[2,8],[2,1],[2,2]),
 'weight_dtype':('F16',[2,8],[2,1],[2,1]),
 'zero_rows':('U32',[0,8],[0,1],[0,1]),
 'overflow':('U32',[0,8],[0,1<<58],[0,1<<58]),
 'stacked':('U32',[2,3,8],[2,3,1],[2,3,1]),
}
for name,(dtype,weight,scale,bias) in cases.items():
 folder=root/name;folder.mkdir(exist_ok=True);t={}
 for key in ['mtp.pre_fc_norm_embedding.weight','mtp.pre_fc_norm_hidden.weight','mtp.norm.weight']:t[key]=('BF16',[1])
 t.update({'mtp.fc.weight':(dtype,weight),'mtp.fc.scales':('BF16',scale),'mtp.fc.biases':('BF16',bias)})
 h={};raw=bytearray()
 for key,(dt,shape) in t.items():
  count=1
  for v in shape:count*=v
  size=count*({'U32':4,'BF16':2,'F16':2}[dt]);h[key]={'dtype':dt,'shape':shape,'data_offsets':[len(raw),len(raw)+size]};raw.extend(b'\0'*size)
 hdr=json.dumps(h).encode();(folder/'mtp.safetensors').write_bytes(struct.pack('<Q',len(hdr))+hdr+raw)
print(root)
