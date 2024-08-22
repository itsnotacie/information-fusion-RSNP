import matplotlib.pyplot as plt

# K = [3,4,5,6,7]
# ACC= [93.78, 93.80, 93.83, 93.84, 93.76]
# plt.plot(K, ACC, marker='*')
# plt.xlabel('K')
# plt.yticks([50, 60, 70, 80,90, 100])
# plt.xticks([3,4,5,6,7])
# plt.ylabel('Acc')
# plt.savefig('K-Acc.pdf')
# plt.show()

Acc = [92.4, 93.1, 93.83, 93.84, 93.80]
params = [0.348, 0.353, 0.372, 0.376, 0.395]
# flops = [50.4, 55.2, 60.0, 64.8, 69.5]
layer_num = [12, 13, 14,15,16]

fig, ax1 = plt.subplots()

# Plot Accuracy vs. Number of Layers with y-axis color set to blue
ax1.plot(layer_num, Acc, 'b-o', label='Acc (%)')
ax1.set_xlabel('Number of Layers')
ax1.set_xticks([12, 13, 14, 15, 16])
# Make y-axis label and line color match
ax1.set_ylabel('Acc (%)', color='b')
ax1.set_yticks([92.0, 92.4, 92.8, 93.2, 93.6, 94.0])
ax1.tick_params('y', colors='b')

# Create a twin Axes sharing the x-axis
ax2 = ax1.twinx()
# Plot Params and FLOPs vs. Number of Layers
ax2.plot(layer_num, params, 'g-^', label='Params (M)')
# ax2.plot(layer_num, flops, 'g-s', label='FLOPs (Billion)')
ax2.set_ylabel('Params', color='g')
# Set the color of the right y-axis labels
ax2.tick_params('y', colors='g')

# Add a legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.savefig('layer deletion.pdf')
plt.show()