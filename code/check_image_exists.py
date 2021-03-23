# check whether each of the images used here is contained in the downloaded entirety of images
import os
path = "../16-class-ImageNet/image_names"
txt_file_list = os.listdir(path)
print(txt_file_list)
for txt_file in txt_file_list:
    file_path = os.path.join(path, txt_file)
    print(file_path)
    print(f"Now scanning {file_path}")
    no_of_existing_files = 0
    lst_not_found_images = []

    line_count = 0
    with open(file_path) as f:
        for img in f:
            if img != "\n":
                line_count += 1
            img = img.strip()
            # print(img)
            location1 = "../../ILSVRC2012_img_train/"
            location2 = "../../ILSVRC2012_img_train_t3/"
            if os.path.isfile(location1 + img):
                no_of_existing_files += 1
        #         print(f"{image_name} has been found.")
            elif os.path.isfile(location2 + img):
                no_of_existing_files += 1
            else:
                lst_not_found_images.append((img, f))

    print("no_of_existing_files", no_of_existing_files, f"in the file with {line_count} lines")
    print(len(lst_not_found_images))

        

                                                                                                                            
