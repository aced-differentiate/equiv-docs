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
x=[1 2; 3 4]/10
Gray.(x)