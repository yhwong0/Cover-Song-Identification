import os
import pickle
import numpy as np
import libfmp.c3

audio_dir = "/research/d2/fyp23/tcor0/train_ori01"
temp_class_list = os.listdir(audio_dir)
class_list = []
for i in range(len(temp_class_list)):
    try:
        if os.listdir(audio_dir + "/" + temp_class_list[i]) != []:
            class_list.append(temp_class_list[i])
    except:
        pass

# print(class_list)
# class_list.remove("0")
class_list_len = len(class_list)
class_label = [i for i in range(class_list_len)]
mapping = dict(zip(class_list, class_label))
# print(mapping)
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(mapping, f)
    f.close()
print("mapping saved into saved_dictionary.pkl, " + str(class_list_len) + " classes in total", )
count = 0
cqt_shape_sum = 0
threshold_cqt_len = 20000
total_cqt_mean = []
for classes in class_list:
    # print(os.listdir(audio_dir + "/" + classes))
    wav_dir = audio_dir + "/" + classes + "/"
    try:
                for file in os.listdir(audio_dir + "/" + classes):
                    if file.endswith(".npy"):
                        if os.path.getsize(audio_dir + "/" + classes + "/" + file) > 0:
                            cqt = np.load(audio_dir + "/" + classes + "/" + file, allow_pickle=True)
                            #cqt_shape_sum = cqt_shape_sum + cqt.shape[1]
                            total_cqt_mean.append(np.mean(cqt))
                    else:
                        raise IOError("Incorrect file type!")

                    # print(cqt.shape)
                    count += 1

    except IOError:
                pass
    except PermissionError:
                pass
    except EOFError:
                pass
    except pickle.UnpicklingError:
                pass
    except pickle.UnpicklingError:
                pass

print("count ", count)
print(len(total_cqt_mean))
with open('cqtmean.npy', 'wb') as f:
    np.save(f, np.array(total_cqt_mean))
    f.close()
print("mean",np.mean(total_cqt_mean))
print("standard deviation",np.std(total_cqt_mean))
#print("total sum", cqt_shape_sum)
#print("avg length:", cqt_shape_sum / count)

