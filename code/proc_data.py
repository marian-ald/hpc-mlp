import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 


def normalize_data():
    df = pd.read_csv('../data/heart_disease_.csv', delimiter=',')

    normalized_df=(df-df.min())/(df.max()-df.min())

    normalized_df.to_csv('../data/heart_disease_norm.csv', index=False)



def comp_histogram():

    data = [0.0544085, 0.146682, 0.419363, -0.423572, -0.41971, 0.413888, 0.187122, -0.144257, 0.324158, -0.166224, 0.391933, -0.249237, 0.098217, 0.278275, -0.0946493, 0.102554, 0.276693, -0.446796, 0.142452, 0.363217, 0.345601, -0.024762, -0.143468, 0.181301, 0.26442, 0.172894, -0.166855, 0.296786, -0.260206, 0.45132, -0.227401, 0.257112, 0.135091, -0.270948, 0.29645, 0.178291, -0.31997, 0.0206623, -0.428876, -0.458722, 0.317348, 0.425967, -0.245049, -0.0473445, 0.241332, 0.123212, -0.4077, 0.055115, 0.139326, 0.197661, -0.0445778, 0.0220168, -0.290011, 0.274864, -0.259593, 0.437319, -0.0151519, 0.0364627, 0.271195, 0.187552, 0.0248722, -0.419115, -0.0182462, -0.302947, -0.227153, -0.184706, 0.338254, -0.0842132, 0.298866, 0.372289, -0.0800252, 0.153305, 0.335346, 0.137836, -0.35695, 0.113768, -0.201862, -0.30174, -0.294027, 0.400373, 0.358831, 0.124305, -0.0405198, -0.39409, -0.0637405, 0.162798, -0.419681, 0.384018, -0.26365, 0.314425, 0.108659, 0.224133, 0.358219, -0.372497, 0.384096, -0.331844, -0.0942928, 0.25944, 0.0468528, -0.258336, 0.168819, 0.429738, 0.357878, 0.0412544, 0.104663, -0.461981, -0.307888, 0.365711, -0.300811, -0.139005, 0.303175, -0.40489, 0.448211, -0.200255, -0.33607, -0.07844, 0.425452, -0.292841, -0.157332, -0.301107, -0.441326, 0.414237, 0.385935, 0.379803, -0.42117, 0.307121, -0.414951, -0.0525529, 0.103651, 0.0948118, 0.152021, -0.19044, 0.0616393, 0.0469892, 0.313724, -0.296607, 0.0479179, -0.457073, -0.393806, 0.210017, -0.133168, 0.372279, 0.268037, -0.147868, -0.290887, 0.394877, 0.236603, -0.328344, -0.360874, -0.38364, -0.166541, -0.33929, -0.432313, -0.243516, -0.422397, -0.390573, -0.399305, -0.374438, 0.0197841, 0.167256, 0.183284, -0.291105, 0.439726, -0.217987, 0.218794, 0.29054, -0.0516843, -0.196198, 0.296377, 0.0174197, -0.449092, -0.299701, -0.0732117, 0.281855, 0.0153411, 0.0988118, 0.213822, -0.210966, 0.233378, 0.315858, -0.131696, -0.396074, 0.439478, -0.101099, -0.17668, -0.44583, -0.0287623, -0.113075, -0.357358, 0.453932, -0.408729, 0.288836, -0.300084, -0.431913, -0.392061, 0.38162, 0.321537, 0.0191648, -0.277488, 0.155003, -0.426326, -0.263669, 0.318212, -0.0366272, -0.444724, -0.129357, -0.400725, 0.232007, 0.122587, 0.295562, 0.0670134, 0.357963, 0.285864, 0.0485296, -0.086934, -0.218651, 0.0620029, 0.255526, 0.057302, 0.145264, 0.248444, 0.10004, 0.00795244, -0.353413, 0.124491, 0.063839, 0.312761, 0.0129739, -0.286192, -0.271272, -0.229906, -0.257334, -0.114109, -0.344046, 0.0789227, -0.099764, -0.0809355, 0.127975, -0.281902, -0.349386, -0.00403105, 0.150259, -0.356572, -0.0833155, -0.166359, -0.0783577, 0.0631814, 0.260792, -0.18798, -0.244665, 0.0409072, -0.304684, 0.220524, -0.316289, -0.292949, -0.0201338, 0.112699, -0.345336, 0.357989, 0.191655, -0.25146, -0.237066, 0.29947, -0.000420481, -0.215963, 0.0132438, 0.264964, 0.0682495, -0.22393, 0.34821, 0.0840117, 0.137187, 0.133321, 0.092588, -0.311277, 0.33211, -0.350918, 0.117053, 0.227753, -0.173749, 0.237537, -0.0964881, -0.113285, 0.0929121, -0.0476285, -0.0410854, -0.29237, -0.300078, -0.0212729, -0.29953, 0.256725, 0.0924157, -0.171448, 0.191046, -0.273153, -0.0222625, -0.160858, 0.356959, -0.319161, -0.0196399, 0.340021, 0.129999, -0.247601, 0.108193, -0.142562, -0.193729, 0.0751543, -0.128331, 0.288473, -0.062241, 0.0630675, 0.160861, 0.206419, 0.314931, -0.111375, -0.206358, -0.0913028, -0.038597, -0.141287, 0.252573, 0.0270215, -0.249711, -0.02016, 0.220722, 0.306484, 0.0718353, -0.166689, -0.219523, 0.0636461, -0.120701, 0.125986, 0.0385183, -0.355851, 0.243533, -0.218437, -0.133265, -0.315345, 0.221866, 0.103552, 0.338276, -0.205524, -0.198528, 0.133989, -0.364253, -0.248746, -0.342535, -0.205462, 0.0250997, -0.0159836, 0.0183987, -0.087476, -0.354111, 0.133836, 0.257512, 0.23176, 0.0751718, -0.0358007, -0.300077, 0.220797, -0.337303, -0.0556299, -0.0183654, 0.0663637, -0.0463327, -0.139981, 0.213075, 0.185551, -0.090177, 0.0697923, -0.0760447, -0.117049, 0.229417, 0.0905752, -0.348209, 0.230312, 0.206978, -0.325595, -0.340298, -0.133071, 0.0235696, 0.0432489, 0.144601, 0.0346075, -0.188063, 0.0369655, -0.0987807, 0.252257, -0.363984, -0.0337091, 0.107906, -0.336138, 0.275809, -0.275608, 0.0953739, -0.135672, -0.0504403, -0.0566994, -0.315269, 0.224531, -0.352056, -0.0261653, -0.257666, 0.24251, -0.300738, -0.240726, 0.107674, 0.271388, -0.201173, 0.132524, -0.226832, 0.187545, -0.189376, 0.282918, -0.142996, -0.0122906, -0.0452646, 0.123372, -0.125182, -0.0440999, -0.275486, 0.347872, -0.0150897, -0.364825, -0.292884, -0.284864, -0.135348, 0.0218243, 0.0235848, -0.0854689, -0.118793, 0.0366777, 0.253514, -0.0113109, -0.085961, 0.317924, 0.113111, -0.343436, 0.224164, 0.277086, 0.154236, 0.36248, 0.0994826, 0.330009, 0.28025, 0.321635, -0.04743, -0.130163, 0.0798579, 0.192536, 0.190886, 0.16952, 0.17526, -0.189352, 0.169844, 0.247525, -0.109068, -0.330653, -0.0957989, 0.279665, -0.0509735, 0.150557, -0.0488054, -0.162608, -0.225903, 0.230382, -0.209832, 0.252357, 0.252095, -0.350817, 0.164294, 0.0411825, -0.353484, -0.101371, 0.00604308, 0.291914, -0.144885, 0.323761, -0.203397, 0.300121, 0.151149, 0.352637, 0.104493, -0.0387384, -0.201863, -0.0908113, -0.156362, 0.0542171, -0.056316, 0.112988, -0.031266, 0.257859, -0.101604, 0.285077, -0.269897, 0.0376416, 0.150311, -0.11458, -0.0751501, 0.0372567, -0.100249, -0.276004, -0.286709, -0.0885847, -0.012227, 0.0844824, -0.161819, 0.208036, 0.0430954, -6.73658e-05, 0.14301, -0.170903, -0.0125784, -0.117645, 0.155506, 0.150707, 0.156692, 0.364293, -0.160225, -0.264773, 0.112133, 0.173658, 0.358234, -0.35462, 0.0935864, -0.276811, 0.0481701, -0.121251, -0.0262431, 0.338168, 0.281154, 0.238657, -0.302984, 0.359593, -0.215076, 0.0499373, 0.0789269, -0.0117469, -0.107175, -0.243126, 0.353334, -0.329313, -0.0488812, -0.0243927, -0.0818105, -0.258523, -0.238834, -0.290267, -0.259378, -0.0339106, -0.189892, 0.217903, -0.225401, -0.196806, 0.228431, 0.233334, -0.108468, -0.0885471, -0.253066, 0.230437, -0.115527, -0.337061, 0.103945, -0.0533628, -0.342616, 0.254017, 0.361723, 0.101459, -0.122878, -0.1106, 0.223481, -0.134692, -0.074765, -0.190548, 0.206064, 0.208573, -0.0839229, 0.332377, 0.283454, 0.0218473, -0.0666814, -0.271586, -0.125398, 0.0730658, -0.103244, -0.262116, -0.0587489, 0.153437, 0.0144857, 0.0533333, 0.0187254, 0.264107, 0.0814208, -0.242478, -0.154404, 0.103953, -0.353609, -0.15783, -0.159736, -0.111338, 0.0967189, -0.301403, 0.119118, -0.343195, -0.126803, -0.0399668, 0.230527, 0.154422, -0.0727377, 0.148832, -0.188879, 0.225729, 0.242394, 0.0508711, -0.0663533, -0.225998, 0.153904, 0.240046, 0.292587, -0.196759, -0.0717689, -0.0538356, -0.2978, -0.355497, 0.0688352, -0.0870558, 0.113605, 0.0803749, 0.120263, 0.319017, 0.334185, -0.148166, -0.347535, 0.0881549, -0.126213, -0.10919, -0.31696, -0.260834, -0.319916, -0.0245496, 0.253146, -0.143647, -0.163969, 0.130392, 0.272373, 0.134826, 0.269543, 0.0611283, 0.00972412, 0.196982, 0.229518, 0.303104, -0.222002, 0.296866, 0.312755, 0.211981, -0.155338, 0.0612118, -0.0727924, 0.330073, 0.0150801, -0.103756, -0.183241, 0.0326932, 0.349548, 0.0556944, 0.288652, -0.332561, 0.160008, 0.333884, 0.00803783, 0.0480061, -0.174911, 0.209218, -0.18675, -0.267687, -0.0211045, -0.282356, 0.15859, 0.353768, 0.279774, 0.0229593, 0.291723, -0.307377, -0.0453229, 0.23933, 0.269752, 0.164487, -0.0646064, -0.168188, 0.129413, 0.315622, 0.0932043, 0.31132, -0.0168331, 0.0776036, 0.00186559, -0.09333, 0.110191, -0.203274, -0.124595, -0.246919, 0.20988, 0.0656424, 0.327446, -0.342019, 0.163104, -0.0588065, -0.259227, -0.0434549, -0.0701869, -0.344601, 0.344653, -0.143612, -0.28683, -0.0658185, -0.26943, 0.348071, -0.266479, 0.0311117, -0.185266, 0.228082, -0.0184146, 0.273086, 0.174253, 0.329901, -0.0144583, -0.18903, -0.128578, -0.269416, -0.027156, 0.111976, -0.151187, -0.182424, -0.18753, -0.188889, -0.159295, 0.340722, 0.117453, -0.053373, -0.067881, -0.317882, -0.0328259, -0.0883766, -0.0963458, 0.0454923, 0.210953, -0.000627618, 0.0284145, 0.309622, -0.334664, 0.208297, 0.172555, 0.0120695, 0.116235, -0.0183401, -0.0231782, -0.263372, 0.157778, 0.213393, -0.167639, -0.234526, -0.0397798, 0.0463222, -0.0518022, 0.137839, 0.222582, 0.154051, 0.113412, -0.0251134, -0.26447, -0.319617, 0.0221529, 0.0678525, -0.0428452, 0.290956, -0.251804, -0.19704, -0.0748205, 0.141759, -0.252566, -0.0443363, -0.0150925, 0.285137, 0.332882, -0.264006, -0.098351, -0.055445, -0.16223, -0.305721, -0.207201, 0.0352789, -0.175099, 0.118168, -0.283547, 0.138247, -0.109142, 0.304183, -0.0728498, -0.360878, -0.0860788, 0.0278286, -0.315346, 0.301222, -0.269467, 0.00695674, 0.22703, -0.156122, 0.175065, -0.212939, 0.350785, 0.287647, 0.107873, -0.0294557, 0.207636, 0.0756061, 0.0716865, -0.255863, -0.344987, 0.274605, -0.196436, -0.18704, -0.0552646, -0.00638687, 0.296276, 0.0263365, -0.233288, -0.178014, -0.0346289, 0.0590105, -0.173743, 0.244441, -0.278309, -0.123942, 0.180515, -0.182628, 0.248164, 0.0423961, 0.0263977, 0.05808, 0.194605, 0.0120345, -0.0194215, -0.0626702, 0.347727, -0.176934, -0.352213, 0.0542653, -0.067649, -0.332051, -0.0362782, 0.101063, -0.153943, 0.273606, -0.270472, -0.222815, -0.0652063, -0.138612, -0.0356804, 0.265313, 0.285547, 0.155724, 0.144606, -0.357911, -0.333365, -0.040028, -0.17539, 0.279946, -0.36278, 0.216156, -0.027122, 0.196973, -0.136958, 0.318605, -0.230845, -0.154379, -0.223477, -0.217909, 0.265035, 0.0740219, -0.184813, -0.136392, -0.190063, 0.026393, -0.227935, -0.0953871, 0.168727, 0.0720072, 0.131149, -0.232102, -0.027828, 0.0515481, 0.288771, -0.248371, 0.058786, 0.320554, 0.0767495, 0.248544, 0.235352, 0.0791175, 0.0995514, -0.156919, -0.0890575, 0.327742, -0.203462, 0.0452456, -0.191786, -0.0617913, 0.192485, -0.2919, -0.352918, -0.357476, -0.0631433, -0.177833, 0.034065, 0.0740702, 0.0919286, -0.162357, -0.219071, -0.14207, -0.0293106, 0.118249, 0.274626, -0.105688, 0.235027, -0.0317363, -0.150283, -0.0533719, -0.148341, -0.28008, -0.339403, 0.316359, -0.0718501, -0.0633119, 0.278953, 0.0898361, 0.347082, -0.277982, -0.337104, 0.174418, -0.204733, -0.324873, 0.18209, 0.0972723, -0.137557, -0.148993, -0.193806, 0.31952, 0.0537984, -0.0477284, -0.187699, -0.340661, -0.294627, -0.278221, -0.0812004, 0.305548, 0.0551909, 0.133665, -0.112972, 0.271999, 0.218734, -0.0872265, 0.223209, -0.218265, 0.21461, 0.137014, 0.236719, 0.196544, 0.22418, 0.264764, 0.0058136, -0.345701, 0.30504, -0.177245, 0.11672, -0.197666, 0.0389107, 0.288063, -0.243294, -0.272439, -0.124814, -0.0658449, -0.247951, -0.0542931, 0.0210823, 0.0359966, -0.113893, -0.288875, -0.195487, 0.138283, 0.348272, -0.341902, -0.314092, 0.206333, -0.195018, 0.265666, -0.0218016, -0.323447, 0.0970616, -0.162769, 0.306465, -0.262273, -0.143322, 0.246357, -0.0743694, 0.338547, -0.316457, 0.32969, 0.261461, -0.194603, -0.307898, -0.228502, 0.1047, -0.190701, 0.0823534, -0.239366, 0.210444, 0.333608, -0.163092, -0.350191, 0.106743, -0.179969, -0.326945, 0.157799, -0.338784, -0.156815, 0.058317, 0.00456267, -0.115114, -0.20977, 0.206942, -0.173796, -0.106895, -0.301529, -0.292588, 0.183884, -0.32813, -0.243897, 0.148426, 0.298479, -0.0733517, 0.205676, -0.295172, -0.3338, -0.350173, 0.15233, -0.208017, 0.225419, 0.12079, -0.00596121, 0.240376, -0.137615, 0.179218, 0.27858, -0.344965, 0.205583, -0.243383, 0.0785008, -0.155003, 0.00665195, 0.233879, -0.31321, 0.198004, -0.238164, -0.24959, 0.270564, 0.310869, -0.212572, -0.338481, 0.0941462, -0.279242, -0.0466843, -0.0653261, -0.209265, -0.0153357, -0.0503509, 0.308214, 0.141796, -0.19008, 0.0638555, -0.229314, -0.314852, 0.291389, 0.315053, 0.328876, 0.311572, 0.155487, -0.279655, 0.0249249, -0.364665, 0.0921455, -0.106344, -0.312726, -0.0749991, 0.0206404, -0.197168, -0.169583, -0.033639, -0.0445918, -0.142916, -0.304641, 0.0413148, 0.175548, -0.00481892, 0.197198, -0.204936, 0.309979, 0.140263, 0.302008, -0.24525, -0.16103, -0.292454, -0.194953, -0.234789, -0.34255, -0.231225, -0.288365, 0.178085, -0.145732, 0.101708, 0.178569, 0.311562, 0.360512, 0.230991, -0.128585, 0.0160042, -0.331325, 0.0669797, 0.347514, -0.0107689, 0.289212, -0.322276, -0.334602, 0.0996122, 0.0380534, 0.227744, 0.259825, -0.0171163, 0.00285898, 0.196684, 0.102782, 0.206978, 0.269378, 0.272977, 0.337337, 0.291976, -0.323397, -0.316177, 0.104913, -0.10398, 0.15068, -0.0816666, -0.157566, 0.146043, -0.215824, 0.0789965, -0.34061, -0.305225, -0.367563, -0.370185, 0.289087, -0.494912, -0.298285, 0.340314, 0.284516, 0.377905, 0.10988, 0.107882, -0.263173, -0.497698, -0.17464, 0.521571]

    # Creation of plot 
    fig = plt.figure(figsize=(10, 5)) 

    # plotting the Histogram with certain intervals 
    plt.hist(data, bins=50) 
    
    # Giving title to Histogram 
    plt.title("Predefined Histogram") 
    
    # Displaying Histogram 
    plt.show()


def preprocess_iris():
    

comp_histogram()