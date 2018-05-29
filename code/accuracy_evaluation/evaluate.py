"""
Evaluate trained models on image datasets.

An image dataset of a certain experiment is
expected to be structured as follows (exemplary
path of a single image):

path/to/experiment-stimuli/experiment-name/dnn/session-1/0001_hp_dnn_0.4_knife_10_n03041632_8618.png

where session-1/ is one of (potentially many) session directories containing the images to evaluate.
The imagenames are named as described in the README.

"""

import numpy as np
import os
import csv
import sys
import linecache as lc

import human_categories as hc
import predict



def get_WNID_from_index(index):
    """Return WNID given an index of categories.txt"""
    assert(index >= 0 and index < 1000), "index needs to be within [0, 999]"

    filepath = "categories.txt"
    assert(os.path.exists(filepath)), "path to categories.txt wrong!"
    line = lc.getline(filepath, index+1)
    return line.split(" ")[0]


def get_16AFC_decision(probabilities):
    """Return the category that counts as a decision.

    parameters:
    - probabilities: a vector, usually of length 1000. The index
    of the highest value in the vector is counted as the decision
    of a network for a certain category.

    """

    Categories = hc.HumanCategories()
    category_decision = None
    best_probability = -1.0

    for i in range(len(probabilities)):
        if probabilities[i] > best_probability:

            WNID = get_WNID_from_index(i)
            category = Categories.get_human_category_from_WNID(WNID)

            if category is not None:
                category_decision = category
                best_probability = probabilities[i]

    return category_decision


def write_experiment_to_csv(stimulipath,
                            resultdir,
                            experiment_name,
                            dnn_name,
                            network_fun,
                            sort_indices):
    """Evaluate a particular experiment and save results to .csv files."""
    
    session_names = sorted(os.listdir(stimulipath))

    for sess in session_names:
        session_number = int(sess.split("-")[1])
  
        print("Evaluating session {} of {}".format(session_number,
               len(session_names)))

        write_session_to_csv(os.path.join(stimulipath, sess),
                             resultdir,
                             experiment_name,
                             session_number,
                             dnn_name,
                             network_fun,
                             sort_indices)    
                                

def write_session_to_csv(sessionpath,
                         resultdir,
                         experiment_name,
                         session_number,
                         dnn_name,
                         network_fun,
                         sort_indices=None):
    """Evaluate particular session and write results to .csv file."""

    csv_name = experiment_name+"_"+dnn_name+"_session_"+str(session_number)+".csv"
    filepath = os.path.join(resultdir, csv_name)
 
    with open(filepath, "w") as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["subj", "session", "trial",
                         "rt", "object_response", "category",
                         "condition", "imagename"])

        img_list = sorted(os.listdir(sessionpath))

        num_imgs = len(img_list)
        all_images = np.zeros([num_imgs, 224, 224, 3], dtype=np.float32)

        for i, imagename in enumerate(img_list):
            loaded_img = predict.load_image(os.path.join(sessionpath, imagename))
            if len(loaded_img.shape) == 2:
                loaded_img = np.stack([loaded_img, loaded_img, loaded_img],
                                      axis=2)
            all_images[i,:,:,:] = loaded_img
        print("==> Loaded all images")

        _, probabilities, _ = network_fun(all_images)
        print("==> Computed probabilities")

        classnames = predict.ordered_classnames()

        for i, imagename in enumerate(img_list):

            probs = probabilities[i,:]
            if sort_indices is not None:
                probs = probs[sort_indices]
 
            obj_resp = classnames[np.argmax(probs)]

            category = imagename.split("_")[4]
            condition = imagename.split("_")[3]
            writer.writerow([dnn_name, session_number,
                             i+1, "NaN", obj_resp,
                             category, condition, imagename])


def perform_experiment(model_path,
                       stimuli_path,
                       experiment_name,
                       resultdir):
    """Convenience function: call write_experiment_to_csv with sane defaults."""

    assert os.path.exists(model_path)
    assert os.path.exists(stimuli_path)

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    write_experiment_to_csv(stimulipath = stimuli_path,
        resultdir = resultdir,
        experiment_name = experiment_name,
        dnn_name = model_path.split("/")[-1].replace("_", "-"),
        network_fun = lambda y: predict.predict(model_path, y),
        sort_indices = None)


def main():

    # Step 1: define directory paths

    # define path to model checkpoints. Within checkpoints/ , every
    # model should have its own directory.
    model_dir = "/gpfs01/bethge/home/rgeirhos/shared_from_jonas/checkpoints/"

    # the directory where every experiment has its own subdirectory (structure
    # described on the top of this file=
    experiment_dir = "/gpfs01/bethge/home/rgeirhos/experiment-stimuli/"
 
    # the directory where all results should be stored
    resultdir = "/gpfs01/bethge/home/rgeirhos/object-recognition-combined/raw-data/fine-tuning/"


    # Step 2: run evaluation

    evaluate_all_models = False
    if evaluate_all_models:
        model_list = []
        for i, m in enumerate(sorted(os.listdir(model_dir))):
            if i >= 0: # set this to N if first N models need to be excluded
                print("Added to list of models: "+m)
                model_list.append(m)
    else:
        # if only one particular model should be evaluated
        model_list = ["sixteen35__phase_scrambling_multiple__uniform_noise_multiple__200epochs"]

    print(model_list)
    if "sixteen13__color__grayscale_contrast_multiple__uniform_noise_multiple__resnet152" in model_list:
        print("model sixteen13 has a ResNet-152 architecture: sure this should be evaluated?")
        sys.exit(0)

    experiment_list = []
    for e in sorted(os.listdir(experiment_dir)):
        print("Added to list of experiments: "+e)
        experiment_list.append(e)
    print(experiment_list)


    num_runs = len(model_list) * len(experiment_list)
    counter = 0
    for m in model_list:
        for e in experiment_list:
            print("Run #"+str(counter+1)+" of "+str(num_runs))
            print("Model "+m+"; experiment "+e)
            perform_experiment(model_path = os.path.join(model_dir, m),
                               stimuli_path = os.path.join(experiment_dir, e, "dnn/"),
                               experiment_name = e,
                               resultdir = os.path.join(resultdir, e)) 
            counter += 1


if __name__ == "__main__":

    main()
