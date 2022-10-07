# using Plots
# using Images
# x=collect(train_x)[2]
# c=colorview(RGB,[x[:,:,i] for i=1:3]...)
# g=Gray.(c)
# # x=[RGB(r,g,b) for (r,g,b) in zip(eachslice(x,dims=3)...)]
# # x=sum(abs2,x,dims=3)[:,:,1]
# mosaicview(g,c,nrow=1)
# # heatmap(x)
#     # x=detect_edges(x, canny)
# x=[1 2; 3 4]/10
# Gray.(x)

img=train_x[:,:,1]
img=imrotate(img, 0)
imgr=imrotate(img, π/2)

# imgr=cat(imrotate.(eachslice(img,dims=3), π/2)...,dims=3)
# img=train_x[:,:,:,1]
# img=cat(imrotate.(eachslice(img,dims=3), 0)...,dims=3)
# imgr=cat(imrotate.(eachslice(img,dims=3), π/2)...,dims=3)
# img=reshape(img,(size(img)...,1))
# imgr=reshape(imgr,(size(imgr)...,1))

# norm(feat(img)-feat(imgr))
mean(abs.(feat(img)-feat(imgr)))

# img=img[:,:,1]
# v= [op(img)[1] for op in ops]
# imgr=imgr[:,:,1]
# vr= [op(imgr)[1] for op in ops]

"""
