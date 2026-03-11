from rigno import dataset_Heat3D

datadir = "/home/xyh/myCode/rigno-main/dataset_3d_heat"

dataset = dataset_Heat3D.Heat3DDataset(datadir)

print("\nDataset size:")
print(len(dataset))


print("\nLoad single sample")
u,x,c = dataset[0]

print("u shape:",u.shape)
print("x shape:",x.shape)
print("c shape:",c.shape)


print("\nLoad batch")

batch_indices=[0,1]

u,x,c=dataset.get_batch(batch_indices)

print("batch u:",u.shape)
print("batch x:",x.shape)
print("batch c:",c.shape)