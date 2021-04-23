cellfinder -s green_channel -b red_channel -o ./cellfinder -v 25 4.22 4.1 --orientation psl --threshold 20 --start-plane 100 --end-plane 250 --trained-model /home/guido/Histology/cellfinder_trained_model/model.h5
cellfinder_curate green_channel red_channel ./cellfinder/points/cell_classification.xml --v 25 4 4
cellfinder_train -y yaml_1.yml -o /path/to/output/directory/

