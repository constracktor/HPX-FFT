import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plot_general import plot_object
from plot_general import colors_16 as colors
from plot_general import greyscale
from plot_general import plot_bar
from plot_general import extract_fft_sections
from plot_general import extract_comm
from plot_general import extract_comm_error
# Configure font and fontsize
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams.update({'errorbar.capsize': 5})
# set number of runs
n_loop = 50
################################################################################
# create print objects for mstrong scaling on buran
ss_fftw_mpi_threads = plot_object('./plot/data/strong_scaling/buran_parcelports/strong_runtimes_fftw_mpi_threads.txt', n_loop)

scatter_buran_mpi = plot_object('./plot/data/parcelport/strong_scaling_buran/scatter_mpi.txt', 50)
scatter_buran_lci = plot_object('./plot/data/parcelport/strong_scaling_buran/scatter_lci.txt', 50)
scatter_buran_tcp = plot_object('./plot/data/parcelport/strong_scaling_buran/scatter_tcp.txt', 50)

all_to_all_buran_mpi = plot_object('./plot/data/parcelport/strong_scaling_buran/all_to_all_mpi.txt', 50)
all_to_all_buran_lci = plot_object('./plot/data/parcelport/strong_scaling_buran/all_to_all_lci.txt', 50)
all_to_all_buran_tcp = plot_object('./plot/data/parcelport/strong_scaling_buran/all_to_all_tcp.txt', 50)

################################################################################
# create print objects for message scaling on buran
message_buran_mpi = plot_object('./plot/data/parcelport/message_buran/message_mpi.txt', 50)
message_buran_lci = plot_object('./plot/data/parcelport/message_buran/message_lci.txt', 50)
message_buran_tcp = plot_object('./plot/data/parcelport/message_buran/message_tcp.txt', 50)
#message_buran_shmem = plot_object('./plot/data/parcelport/message_buran/message_shmem.txt', 1)
#message_fftw = plot_object('./plot/data/parcelport/message_buran/message_fftw.txt', 1)

# # create print objects for message scaling on medusa
# message_medusa_mpi = plot_object('./plot/data/parcelport/message_medusa/message_mpi.txt', 1)
# message_medusa_lci = plot_object('./plot/data/parcelport/message_medusa/message_lci.txt', 1)
# message_medusa_tcp = plot_object('./plot/data/parcelport/message_medusa/message_tcp.txt', 1)
# #message_medusa_shmem = plot_object('./plot/data/parcelport/message_medusa/message_shmem.txt', 1)

################################################################################
# Strong scaling runtime on buran for scatter collective 
plt.figure(figsize=(7,6))
plt.grid()
points = np.array([1,2,4,8,16])
# # ideal scaling
# ideal = 2 * ss_fftw_mpi_threads.median[1,6]/ points
# plt.plot(points, ideal, '--', c=greyscale[2], linewidth=1.5)
# error bars
plt.errorbar(points, ss_fftw_mpi_threads.mean[:,6], yerr = ss_fftw_mpi_threads.confidence_error(6), fmt='s-', c=colors[0], linewidth=2, label='FFTW3 reference')
plt.errorbar(points, scatter_buran_tcp.mean[:,7], yerr = scatter_buran_tcp.confidence_error(7), fmt='o-', c=colors[11], linewidth=2, label='HPX TCP scatter')
plt.errorbar(points, scatter_buran_mpi.mean[:,7], yerr = scatter_buran_mpi.confidence_error(7), fmt='o-', c=colors[4], linewidth=2, label='HPX MPI scatter')
plt.errorbar(points, scatter_buran_lci.mean[:,7], yerr = scatter_buran_lci.confidence_error(7), fmt='o-', c=colors[2], linewidth=2, label='HPX LCI scatter')


# plot parameters
plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
plt.xlabel('N nodes')
plt.xscale("log")
labels_x = ['1','2','4','8','16']
plt.xticks(ticks=points, labels= labels_x)
plt.yscale("log")
plt.yticks(ticks=[0.1, 1.0, 10.0,100.0])
plt.ylabel('Runtime in s')
plt.savefig('plot/figures/strong_scaling_buran_parcelport_scatter_runtime.pdf', bbox_inches='tight')

################################################################################
# Strong scaling runtime on buran for all_to_all collective 
plt.figure(figsize=(7,6))
plt.grid()
points = np.array([1,2,4,8,16])
# # ideal scaling
# ideal = 2 * ss_fftw_mpi_threads.median[1,6]/ points
# plt.plot(points, ideal, '--', c=greyscale[2], linewidth=1.5)

# error bars
plt.errorbar(points, ss_fftw_mpi_threads.mean[:,6], yerr = ss_fftw_mpi_threads.confidence_error(6), fmt='s-', c=colors[0], linewidth=2, label='FFTW3 reference')
plt.errorbar(points, all_to_all_buran_tcp.mean[:,7], yerr = all_to_all_buran_tcp.confidence_error(7), fmt='o-', c=colors[11], linewidth=2, label='HPX TCP all_to_all')
plt.errorbar(points, all_to_all_buran_mpi.mean[:,7], yerr = all_to_all_buran_mpi.confidence_error(7), fmt='o-', c=colors[4], linewidth=2, label='HPX MPI all_to_all')
plt.errorbar(points, all_to_all_buran_lci.mean[:,7], yerr = all_to_all_buran_lci.confidence_error(7), fmt='o-', c=colors[2], linewidth=2, label='HPX LCI all_to_all')


# plot parameters
plt.legend(bbox_to_anchor=(1.0, 0), loc="lower right")
plt.xlabel('N nodes')
plt.xscale("log")
labels_x = ['1','2','4','8','16']
plt.xticks(ticks=points, labels= labels_x)
plt.yscale("log")
plt.yticks(ticks=[0.1, 1.0, 10.0])
plt.ylabel('Runtime in s')
plt.savefig('plot/figures/strong_scaling_buran_parcelport_all_to_all_runtime.pdf', bbox_inches='tight')


################################################################################
# Strong scaling runtime on medusa for scatter collective 
plt.figure(figsize=(7,6))

#TODO

plt.xlabel('N nodes')
plt.xscale("log")
labels_x = ['1','2','4','8','16']
plt.xticks(ticks=points, labels= labels_x)
plt.yscale("log")
plt.yticks(ticks=[0.1, 1.0, 10.0,100.0])
plt.ylabel('Runtime in s')
plt.savefig('plot/figures/strong_scaling_medusa_parcelport_scatter_runtime.pdf', bbox_inches='tight')

################################################################################
# Strong scaling runtime on buran for all_to_all collective 
plt.figure(figsize=(7,6))

#TODO

plt.xlabel('N nodes')
plt.xscale("log")
labels_x = ['1','2','4','8','16']
plt.xticks(ticks=points, labels= labels_x)
plt.yscale("log")
plt.yticks(ticks=[0.1, 1.0, 10.0,100.0])
plt.ylabel('Runtime in s')
plt.savefig('plot/figures/strong_scaling_medusa_parcelport_all_to_all_runtime.pdf', bbox_inches='tight')



################################################################################
# Message scaling runtime on buran for scatter collective 
plt.figure(figsize=(7,6))
# ideal scaling
ticks_x = np.linspace(1,8,8)
labels_x = ['$2^{13}$','$2^{15}$','$2^{17}$','$2^{19}$','$2^{21}$','$2^{23}$','$2^{25}$','$2^{27}$']

# error bars
plt.errorbar(ticks_x, extract_comm(message_buran_tcp.mean), yerr = extract_comm_error(message_buran_tcp.confidence_error), fmt='v-', c=colors[11], linewidth=2, label='TCP parcelport')
plt.errorbar(ticks_x, extract_comm(message_buran_mpi.mean), yerr = extract_comm_error(message_buran_mpi.confidence_error), fmt='v-', c=colors[4], linewidth=2, label='MPI parcelport')
plt.errorbar(ticks_x, extract_comm(message_buran_lci.mean), yerr = extract_comm_error(message_buran_lci.confidence_error), fmt='v-', c=colors[2], linewidth=2, label='LCI parcelport')

# plot parameters
plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
plt.xlabel('Message size (in double-precision floating-point numbers)')
plt.xticks(ticks=ticks_x, labels= labels_x)
plt.yticks(ticks=[0.001, 0.01, 0.1, 1.0, 10.0])
plt.yscale("log")
plt.ylabel('Communication time in s')
plt.savefig('plot/figures/strong_scaling_buran_parcelport_message_comm.pdf', bbox_inches='tight')

################################################################################
# Message scaling runtime on buran for scatter collective 
plt.figure(figsize=(7,6))
matplotlib.rcParams.update({'errorbar.capsize': 2})
# plot details
bar_width = 0.25
epsilon = .015
ticks_x = np.linspace(1,8,8)
labels_x = ['$2^{13}$','$2^{15}$','$2^{17}$','$2^{19}$','$2^{21}$','$2^{23}$','$2^{25}$','$2^{27}$']

# error bars
tcp_positions = ticks_x - bar_width
mpi_positions = ticks_x
lci_positions = ticks_x + bar_width

plt.bar(tcp_positions, extract_comm(message_buran_tcp.mean), bar_width-epsilon,
                                yerr =  extract_comm_error(message_buran_tcp.confidence_error),
                                bottom = 0,
                                color=colors[11],
                                edgecolor='black',
                                linewidth=0.5,
                                alpha=1.0,
                                label='TCP parcelport')
plt.bar(mpi_positions, extract_comm(message_buran_mpi.mean), bar_width-epsilon,
                                yerr =  extract_comm_error(message_buran_mpi.confidence_error),
                                bottom = 0,
                                color=colors[4],
                                edgecolor='black',
                                linewidth=0.5,
                                alpha=1.0,
                                label='MPI parcelport')
plt.bar(lci_positions, extract_comm(message_buran_lci.mean), bar_width-epsilon,
                                yerr =  extract_comm_error(message_buran_lci.confidence_error),
                                bottom = 0,
                                color=colors[2],
                                edgecolor='black',
                                linewidth=0.5,
                                alpha=1.0,
                                label='LCI parcelport')

# plot parameters
plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
plt.xlabel('Message size (in double-precision floating-point numbers)')
plt.xticks(ticks=ticks_x, labels= labels_x)
plt.yscale("log")
plt.yticks(ticks=[0.001, 0.01, 0.1, 1.0, 10.0])
plt.ylabel('Communication time in s')
plt.savefig('plot/figures/strong_scaling_buran_parcelport_message_comm_bars.pdf', bbox_inches='tight')

################################################################################
################################################################################                                                
# # Bar plot buran
# plt.figure(figsize=(12,10))   
# # plot details
# bar_width = 0.25
# ticks_x= np.linspace(1,8,8)
# labels_x = ['$2^7$','$2^8$','$2^9$','$2^{10}$','$2^{11}$','$2^{12}$','$2^{13}$','$2^{14}$']
# # section_colors = [colors[9], colors[2], colors[4], colors[8]]
# # section_labels = ['Communication', 'FFT', 'Transpose', 'Rearrange']
# section_colors = [colors[2], colors[4], colors[8], colors[9]]
# section_labels = ['FFT', 'Transpose', 'Rearrange', 'Communication']

# # MPI bar
# mpi_sections = extract_fft_sections(message_buran_mpi.median)
# mpi_bar_positions = ticks_x
# mpi_hatch = '\\'
# # plot bar
# mpi_bar = plot_bar(mpi_bar_positions, mpi_sections, section_colors, section_labels, mpi_hatch, bar_width)
# # create legend
# mpi_legend = plt.legend(mpi_bar, section_labels, title='MPI', bbox_to_anchor=(.25, 1.0), loc="upper left")
# plt.gca().add_artist(mpi_legend)

# # LCI bar
# lci_sections = extract_fft_sections(message_buran_lci.median)
# lci_bar_positions = ticks_x + bar_width
# lci_hatch = ''
# # plot bar
# lci_bar = plot_bar(lci_bar_positions, lci_sections, section_colors, section_labels, lci_hatch, bar_width)
# # create legend
# lci_legend = plt.legend(lci_bar, section_labels, title='LCI', bbox_to_anchor=(.5, 1.0), loc="upper left")
# plt.gca().add_artist(lci_legend)

# # TCP bar
# tcp_sections = extract_comm(message_buran_tcp.mean)
# tcp_bar_positions = ticks_x - bar_width
# tcp_hatch = '///'
# # plot bar
# tcp_bar = plot_bar(tcp_bar_positions, tcp_sections, section_colors[0], section_labels, tcp_hatch, bar_width)
# # create legend

# # SHMEM bar
# shmem_sections = extract_fft_sections(ss_loop_shmem.median)
# shmem_bar_positions = ticks_x - bar_width
# shmem_hatch = '///'
# # plot bar
# shmem_bar = plot_bar(shmem_bar_positions, shmem_sections, section_colors, section_labels, shmem_hatch, bar_width)
# # create legend
# shmem_legend = plt.legend(shmem_bar, section_labels, title='LCI', bbox_to_anchor=(.2, 1.0), loc="upper left")
# plt.gca().add_artist(shmem_legend)

# # plot parameters
# plt.xlabel('tbd')
# #plt.xscale("log")

# plt.xticks(ticks=ticks_x, labels= labels_x)
# plt.yscale("log")
# plt.yticks(ticks=[0.001, 0.01, 0.1, 1.0, 10.0])
# plt.ylabel('Runtime in s')
# plt.savefig('plot/figures/strong_scaling_buran_distribution.pdf', bbox_inches='tight')






# ################################################################################
# ################################################################################                                                
# # Bar plot buran
# plt.figure(figsize=(12,10))   
# # plot details
# bar_width = 0.25
# ticks_x= np.linspace(1,8,8)
# labels_x = ['$2^7$','$2^8$','$2^9$','$2^{10}$','$2^{11}$','$2^{12}$','$2^{13}$','$2^{14}$']
# # section_colors = [colors[9], colors[2], colors[4], colors[8]]
# # section_labels = ['Communication', 'FFT', 'Transpose', 'Rearrange']
# section_colors = [colors[2], colors[4], colors[8], colors[9]]
# section_labels = ['FFT', 'Transpose', 'Rearrange', 'Communication']

# # MPI bar
# mpi_sections = extract_fft_sections(message_buran_mpi.median)
# mpi_bar_positions = ticks_x
# mpi_hatch = '\\'
# # plot bar
# mpi_bar = plot_bar(mpi_bar_positions, mpi_sections, section_colors, section_labels, mpi_hatch, bar_width)
# # create legend
# mpi_legend = plt.legend(mpi_bar, section_labels, title='MPI', bbox_to_anchor=(.25, 1.0), loc="upper left")
# plt.gca().add_artist(mpi_legend)

# # LCI bar
# lci_sections = extract_fft_sections(message_buran_lci.median)
# lci_bar_positions = ticks_x + bar_width
# lci_hatch = ''
# # plot bar
# lci_bar = plot_bar(lci_bar_positions, lci_sections, section_colors, section_labels, lci_hatch, bar_width)
# # create legend
# lci_legend = plt.legend(lci_bar, section_labels, title='LCI', bbox_to_anchor=(.5, 1.0), loc="upper left")
# plt.gca().add_artist(lci_legend)

# # TCP bar
# tcp_sections = extract_fft_sections(message_buran_tcp.median)
# tcp_bar_positions = ticks_x - bar_width
# tcp_hatch = '///'
# # plot bar
# tcp_bar = plot_bar(tcp_bar_positions, tcp_sections, section_colors, section_labels, tcp_hatch, bar_width)
# # create legend
# tcp_legend = plt.legend(tcp_bar, section_labels, title='TCP', bbox_to_anchor=(0, 1.0), loc="upper left")
# plt.gca().add_artist(tcp_legend)

# # # SHMEM bar
# # shmem_sections = extract_fft_sections(ss_loop_shmem.median)
# # shmem_bar_positions = ticks_x - bar_width
# # shmem_hatch = '///'
# # # plot bar
# # shmem_bar = plot_bar(shmem_bar_positions, shmem_sections, section_colors, section_labels, shmem_hatch, bar_width)
# # # create legend
# # shmem_legend = plt.legend(shmem_bar, section_labels, title='LCI', bbox_to_anchor=(.2, 1.0), loc="upper left")
# # plt.gca().add_artist(shmem_legend)

# # plot parameters
# plt.xlabel('tbd')
# #plt.xscale("log")

# plt.xticks(ticks=ticks_x, labels= labels_x)
# plt.yscale("log")
# plt.yticks(ticks=[0.001, 0.01, 0.1, 1.0, 10.0])
# plt.ylabel('Runtime in s')
# plt.savefig('plot/figures/strong_scaling_buran_distribution.pdf', bbox_inches='tight')



################################################################################                                                
# Bar plot medusa

#TODO