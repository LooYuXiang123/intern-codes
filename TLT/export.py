import os
from datetime import datetime
import pytz
from Yolov4 import Yolov4
from FasterRcnn import FasterRcnn
from Ssd import Ssd
import boto3
from botocore.client import Config
from clearml import Task

def get_timestamp():
    timenow = datetime.now(pytz.timezone('Asia/Singapore'))
    return timenow.strftime('%Y-%m-%d-%H:%M:%S')

def create_new_experiment_workspace(model, timestamp):
    # create workspace dir
    timenow = datetime.now(pytz.timezone('Asia/Singapore'))
    workspace_dir = os.path.join('/workspace', 
                                model + '_export_' + timestamp)
    os.mkdir(workspace_dir)

    # create spec dir
    os.mkdir(os.path.join(workspace_dir, 'spec'))

    # create data dir
    os.mkdir(os.path.join(workspace_dir, 'data'))

    # create trained model dir
    os.mkdir(os.path.join(workspace_dir, 'trained_model'))

    return workspace_dir

if __name__ == '__main__':
    # getting AWS info
    bucket = 'dvd'
    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai:443")
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
    # print(AWS_ACCESS_KEY, AWS_SECRET_ACCESS)

    # clearml stuff
    task = Task.init(project_name="DVD-AI", task_name="YOLO Resnet 18 Drone 96 Epochs Export", task_type='custom')
    print ("+")
    task.set_base_docker(f"harbor.dsta.ai/nvidia/tlt_trt7.2.1:v1 --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}")
    print ("++")
    task.execute_remotely(queue_name="queue-1xV100-64ram", exit_process=True)
    print ("+++")

    # available models to choose 
    models = ['yolov4', 'ssd', 'frcnn']

    # specify the model you want to use
    model = 'yolov4'
    
    # specify the files in the object store to be fetched
    spec_subdir = 'od_yolov4_retrain_pruned_2022-01-14-18:27:45/'
    spec_filename = 'yolo_v4_drone_train_resnet18_kitti_further_train.txt'
    trained_model_subdir = 'od_yolov4_retrain_pruned_2022-01-14-18:27:45/weights/'
    trained_model_filename = 'yolov4_resnet18_epoch_096.tlt'
    data_type = 'fp16' # 'fp32' or 'fp16'

    

    if model == 'frcnn':
        zipped_data_filename = 'combine_data_unsplit_resize.zip'
        zipped_tfrecords = 'tfrecords.zip'
        tfrecords_subdir = 'od_frcnn_retrain_pruned_2021-11-24-14:56:41/'


    # check validity of selected model
    assert model in models

    timestamp = get_timestamp()
    WORKSPACE_DIR = create_new_experiment_workspace(model, timestamp)
    DATA_DIR = os.path.join(WORKSPACE_DIR, 'data')
    SPEC_DIR = os.path.join(WORKSPACE_DIR, 'spec')
    TRAINED_MODEL_DIR = os.path.join(WORKSPACE_DIR, 'trained_model')

    s3 = boto3.resource('s3',
                        endpoint_url=AWS_ENDPOINT_URL,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_ACCESS,
                        verify=False)

    print('downloading the spec file and trained model...')
    s3.Bucket(bucket).download_file(os.path.join(trained_model_subdir, trained_model_filename), 
                                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename))
    s3.Bucket(bucket).download_file(os.path.join(spec_subdir, spec_filename),
                                    os.path.join(SPEC_DIR, spec_filename))

    if model == 'frcnn':
        print('Fetching zipped Data and Tfrecords...')
        s3.Bucket(bucket).download_file(zipped_data_filename, os.path.join(WORKSPACE_DIR, zipped_data_filename))
        s3.Bucket(bucket).download_file(os.path.join(tfrecords_subdir, zipped_tfrecords), os.path.join(WORKSPACE_DIR, zipped_tfrecords))
        #unzipping of data file
        print('Unzipping data...')
        if zipped_data_filename.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zipped_data_filename),'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
                DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])
        elif zipped_data_filename.endswith('.tar.gz'): 
            import tarfile
            with tarfile.open(os.path.join(WORKSPACE_DIR, zipped_data_filename),'r') as tar_ref:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar_ref, DATA_DIR)
                DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])
        #unzipping of tfrecords file
        if zipped_tfrecords.endswith('.zip'):
            print('Unzipping tfrecords...')
            import zipfile
            with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zipped_tfrecords), 'r') as zipped_ref:
                zipped_ref.extractall(os.path.join(DATA_DIR, 'tfrecords'))
        elif zipped_tfrecords.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(os.path.join(WORKSPACE_DIR,zipped_tfrecords), 'r') as tarred_ref:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tarred_ref, os.path.join(DATA_DIR,"tfrecords"))

        

    path_to_output_file = os.path.join(WORKSPACE_DIR, os.path.splitext(trained_model_filename)[0] + '_' + data_type + '.etlt')

    if model == 'yolov4':
        #Exporting the trained weights
        yolov4 = Yolov4()
        yolov4.export(os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                    os.path.join(SPEC_DIR, spec_filename),
                    path_to_output_file,
                    data_type)
    elif model == 'ssd':
        #Exporting the trained weights
        ssd = Ssd()
        ssd.export(os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                os.path.join(SPEC_DIR, spec_filename),
                path_to_output_file,
                data_type)

    if model == 'frcnn':
        #Exporting the trained weights
        frcnn = FasterRcnn()
        frcnn.export(os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                os.path.join(SPEC_DIR, spec_filename),
                DATA_DIR,
                path_to_output_file,
                data_type)

    s3_output_subdir = 'od_' + model + '_export_' + timestamp
    s3.Bucket(bucket).upload_file(path_to_output_file, os.path.join(s3_output_subdir, os.path.basename(path_to_output_file)))

    print('Model weights have been uploaded to bucket: ', bucket, 'subdir: ', s3_output_subdir)
