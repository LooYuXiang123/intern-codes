import os
from datetime import datetime
import pytz
from Yolov4 import Yolov4
from FasterRcnn import FasterRcnn
from Ssd import Ssd
import boto3
import itertools
from botocore.client import Config
from clearml import Task

def get_timestamp():
    timenow = datetime.now(pytz.timezone('Asia/Singapore'))
    return timenow.strftime('%Y-%m-%d-%H:%M:%S')

def create_new_experiment_workspace(model, timestamp):
    # create workspace dir
    timenow = datetime.now(pytz.timezone('Asia/Singapore'))
    workspace_dir = os.path.join('/workspace', 
                                model + '_train_' + timestamp)
    os.mkdir(workspace_dir)

    # create spec dir
    os.mkdir(os.path.join(workspace_dir, 'spec'))

    # create data dir
    os.mkdir(os.path.join(workspace_dir, 'data'))

    # create output dir
    os.mkdir(os.path.join(workspace_dir, 'output'))

    # create pretrained model dir
    os.mkdir(os.path.join(workspace_dir, 'pretrained_model'))

    return workspace_dir

if __name__ == '__main__':
    # getting AWS info #object store s3 stuff
    bucket = 'dvd'
    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
    print('AWS_ACCESS_KEY: ', AWS_ACCESS_KEY, 'AWS_SECRET_ACCESS: ', AWS_SECRET_ACCESS)

    # clearml stuff
    task = Task.init(project_name="DVD-AI", task_name="Yolov4 Resnet 18 Drone 100 Epoch New Training (V2)", task_type='training')
    print ("+")
    task.set_base_docker(f"harbor.dsta.ai/nvidia/tltcustom:v2 --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}")
    print ("++")
    task.execute_remotely(queue_name="queue-2xV100-128ram", exit_process=True)
    print ("+++") 

    # available models to choose 
    models = ['yolov4', 'frcnn', 'ssd']

    # specify the model you want to use
    model = 'yolov4'
    training_type = 'retrain_pruned' # 'new_training', 'continue_training' or 'retrain_pruned'

    # specify the files in the object store to be fetched
    zip_data_filename = 'TDDataset.zip'
    spec_subdir = 'specs/' #specs
    spec_filename = 'yolo_v4_drone_train_resnet18_kitti_further_train.txt'
    pretrained_models_subdir = 'od_yolov4_retrain_pruned_2021-11-29-10:23:00/weights/' #pretrained_models/objectdetection
    pretrained_model_filename = 'yolov4_resnet18_epoch_096.tlt' #resnet_18.hdf5

    if model == 'frcnn':
        dataset_config_filename_subdir = 'specs/'
        dataset_config_filename = 'frcnn_drone_config_resnet18_tfrecords.txt'
        #have to set zipped_tfrecords to be '' if tfrecords have yet to be generated for the dataset used to train model
        tfrecords_subdir = ''
        zipped_tfrecords = ''
    elif model == 'ssd':
        # specify the initial_epoch if you wish to continue training
        initial_epoch = 0

    #specify the number of gpu used for training
    no_of_gpus = 2
    gpu_indices = [0, 1]

    #Checking validity of selected model
    assert model in models

    #creating a new environment
    timestamp = get_timestamp()
    WORKSPACE_DIR = create_new_experiment_workspace(model, timestamp)
    SPECS_DIR = os.path.join(WORKSPACE_DIR, 'spec')
    DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
    OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'output')
    PRETRAINED_MODEL_DIR = os.path.join(WORKSPACE_DIR, 'pretrained_model')

    
    s3 = boto3.resource('s3',
                        endpoint_url=AWS_ENDPOINT_URL,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_ACCESS,
                        verify = False)

    print('downloading data, spec file and pretrained_model...')
    s3.Bucket(bucket).download_file(zip_data_filename, os.path.join(WORKSPACE_DIR, zip_data_filename))
    s3.Bucket(bucket).download_file(os.path.join(spec_subdir, spec_filename), os.path.join(SPECS_DIR, spec_filename))
    s3.Bucket(bucket).download_file(os.path.join(pretrained_models_subdir, pretrained_model_filename), 
                                    os.path.join(PRETRAINED_MODEL_DIR, pretrained_model_filename))
    
    # unzip the data file
    print('unzipping...')
    if zip_data_filename.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zip_data_filename),'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])
            print('unzipping complete')
    elif zip_data_filename.endswith('.tar.gz'): 
        import tarfile
        with tarfile.open(os.path.join(WORKSPACE_DIR, zip_data_filename),'r') as tar_ref:
            tar_ref.extractall(DATA_DIR)
            DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])
    
    #Downloading tfrecords and unzipping it
    if model == 'frcnn':
        if zipped_tfrecords:
            print('downloading generated tfrecords...')
            s3.Bucket(bucket).download_file(os.path.join(tfrecords_subdir,zipped_tfrecords), 
                                            os.path.join(WORKSPACE_DIR, zipped_tfrecords))
            
            print('unzippping of tfrecords...')
            if zipped_tfrecords.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zipped_tfrecords), 'r') as zipped_ref:
                    zipped_ref.extractall(os.path.join(DATA_DIR, 'tfrecords'))
            elif zipped_tfrecords.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(os.path.join(WORKSPACE_DIR,zipped_tfrecords), 'r') as tarred_ref:
                    tarred_ref.extractall(os.path.join(DATA_DIR, 'tfrecords'))
    
    #model training command
    if model == 'yolov4':
        yolov4 = Yolov4()
        # start the training 
        yolov4.train(os.path.join(SPECS_DIR, spec_filename), 
                    DATA_DIR, 
                    OUTPUT_DIR, 
                    os.path.join(PRETRAINED_MODEL_DIR, pretrained_model_filename), 
                    no_of_gpus=no_of_gpus, 
                    gpu_indices=gpu_indices,
                    training_type=training_type)

        trained_models_output_dir = os.path.join(OUTPUT_DIR, 'weights')

    elif model == 'frcnn':
        print('fetching dataset config for tfrecords generation...')
        s3.Bucket(bucket).download_file(os.path.join(dataset_config_filename_subdir, dataset_config_filename), os.path.join(SPECS_DIR, dataset_config_filename))
        
        frcnn = FasterRcnn()
        # start the training 
        frcnn.train(os.path.join(SPECS_DIR, dataset_config_filename),
                    os.path.join(SPECS_DIR, spec_filename),
                    DATA_DIR,
                    os.path.join(PRETRAINED_MODEL_DIR, pretrained_model_filename),
                    OUTPUT_DIR,
                    no_of_gpus=no_of_gpus,
                    gpu_indices=gpu_indices,
                    training_type=training_type)

        trained_models_output_dir = os.path.join(OUTPUT_DIR, 'weights')

    elif model == 'ssd':
        ssd = Ssd()

        ssd.train(os.path.join(SPECS_DIR, spec_filename),
                DATA_DIR,
                OUTPUT_DIR,
                os.path.join(PRETRAINED_MODEL_DIR, pretrained_model_filename),
                no_of_gpus=no_of_gpus,
                gpu_indices=gpu_indices,
                initial_epoch=initial_epoch)

        trained_models_output_dir = os.path.join(OUTPUT_DIR, 'weights')

    if training_type == 'retrain_pruned':
        s3_output_subdir = 'od_' + model + '_retrain_pruned_' + timestamp
    else:
        s3_output_subdir = 'od_' + model + '_train_' + timestamp
    
    # upload the spec file used for training
    s3.Bucket(bucket).upload_file(os.path.join(SPECS_DIR, spec_filename), os.path.join(s3_output_subdir, spec_filename))
 
    # upload the weights 
    for path, dirnames, filenames in os.walk(OUTPUT_DIR):
        # extract the path relative to OUTPUT_DIR
        temp = path.split('/')
        idx = temp.index('output')
        temp = [i for i in temp[idx + 1:] if i]
        relative_dir = '/'.join(temp)
        
        for filename in filenames:
            if model == 'frcnn':
                s3.Bucket(bucket).upload_file(os.path.join(path, filename), os.path.join(s3_output_subdir, 'weights', filename))
            else:
                s3.Bucket(bucket).upload_file(os.path.join(path, filename), os.path.join(s3_output_subdir, relative_dir, filename))
    
    #upload the zipped tfrecords for frcnn model
    if model == 'frcnn':
        import shutil
        for folders in os.listdir(DATA_DIR):
            # extracting the TFrecords file and zipping it
            if folders == 'tfrecords':
                tf_records = os.path.join(DATA_DIR, folders)
                print('zipping up')
                zipped_tfrecords = shutil.make_archive(tf_records, 'zip', tf_records)
                zip_tfrecord = os.path.basename(zipped_tfrecords)
                s3.meta.client.upload_file(zipped_tfrecords, bucket, os.path.join(s3_output_subdir, zip_tfrecord))

    print('Model weights have been uploaded to bucket: ', bucket, 'subdir: ', s3_output_subdir)
    print('Data used for training model is ' + zip_data_filename)


