import os
from datetime import datetime
import pytz
from Yolov4 import Yolov4
from FasterRcnn import FasterRcnn
from Ssd import Ssd
import boto3
from clearml import Task

def get_timestamp():
    timenow = datetime.now(pytz.timezone('Asia/Singapore'))
    return timenow.strftime('%Y-%m-%d-%H:%M:%S')

def create_new_prune_workspace(model, timestamp):
    # create workspace dir
    workspace_dir = os.path.join('/workspace', 
                                model + '_prune_' + timestamp)
    os.mkdir(workspace_dir)

    #create spec dir
    os.mkdir(os.path.join(workspace_dir, 'spec'))

    #create trained model dir
    os.mkdir(os.path.join(workspace_dir, 'trained_model'))

    return workspace_dir

if __name__ == '__main__':
    # getting AWS info
    bucket = 'dvd'
    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai:443")
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
    print(AWS_ACCESS_KEY , AWS_SECRET_ACCESS)

    # clearml stuff
    task = Task.init(project_name="DVD-AI", task_name="Frcnn Resnet 18 Drone 82 Epochs Pruning (Syn)", task_type='optimizer')
    print ("+")
    task.set_base_docker(f"harbor.dsta.ai/nvidia/tlt_trt7.2.1:v1 --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}")
    print ("++")
    task.execute_remotely(queue_name="queue-1xV100-128ram", exit_process=True)
    print ("+++")  

    # available models to choose 
    models = ['yolov4', 'frcnn', 'ssd']
    # specify the model you want to use
    model = 'frcnn'

    #check validity of selected model
    assert model in models

    # specify the files in the object store to be fetched
    trained_models_subdir = 'od_frcnn_train_2021-12-03-11:57:19/weights/'
    trained_model_filename = 'frcnn_resnet_18.epoch82.tlt'
    spec_subdir = 'od_frcnn_train_2021-12-03-11:57:19/'
    spec_filename = 'frcnn_drone_train_resnet18_kitti_test_v3.txt'

    if model == 'frcnn':
        tfrecords_subdir = 'od_frcnn_train_2021-12-03-11:57:19/'
        zipped_tfrecords = 'tfrecords.zip'
        

    '''
    Specify the parameters for pruning
    https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/yolo_v4.html#pruning-the-model
    '''

    normalizer = 'max' # max or L2
    equalization_criterion ='union' # arithmetic_mean, geometric_mean, intersection or union
    pruning_granularity = 8
    pth = .1
    min_num_filters = 16
    excluded_layers = [] # [item1, item2]

    timestamp = get_timestamp()
    WORKSPACE_DIR = create_new_prune_workspace(model, timestamp)
    SPEC_DIR = os.path.join(WORKSPACE_DIR, 'spec')
    TRAINED_MODEL_DIR = os.path.join(WORKSPACE_DIR, 'trained_model')

    s3 = boto3.resource('s3',
                        endpoint_url=AWS_ENDPOINT_URL,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_ACCESS,
                        verify=False)

    print('downloading the experiment spec file and the trained model...')
    s3.Bucket(bucket).download_file(os.path.join(trained_models_subdir, trained_model_filename), 
                                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename))
    s3.Bucket(bucket).download_file(os.path.join(spec_subdir, spec_filename), 
                                    os.path.join(SPEC_DIR, spec_filename))

    pruned_model_filename = trained_model_filename.replace('.tlt', '_pruned.tlt')
    if model == 'yolov4':
        yolov4 = Yolov4()
        yolov4.prune(os.path.join(SPEC_DIR, spec_filename),
                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename),   
                    os.path.join(WORKSPACE_DIR, pruned_model_filename),
                    normalizer=normalizer,
                    equalization_criterion=equalization_criterion,
                    pruning_granularity=pruning_granularity,
                    pth=pth,
                    min_num_filters=min_num_filters,
                    excluded_layers=excluded_layers)
    elif model == 'frcnn':
        print('downloading zipped tfrecords...')
        s3.Bucket(bucket).download_file(os.path.join(tfrecords_subdir, zipped_tfrecords), 
                                    os.path.join(WORKSPACE_DIR, zipped_tfrecords))
                                    
        frcnn = FasterRcnn()
        frcnn.prune(os.path.join(TRAINED_MODEL_DIR, trained_model_filename),   
                    os.path.join(WORKSPACE_DIR, pruned_model_filename),
                    normalizer=normalizer,
                    equalization_criterion=equalization_criterion,
                    pruning_granularity=pruning_granularity,
                    pth=pth,
                    min_num_filters=min_num_filters,
                    excluded_layers=excluded_layers)
    elif model == 'ssd':
        ssd = Ssd()
        ssd.prune(os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                os.path.join(WORKSPACE_DIR, pruned_model_filename),
                normalizer=normalizer,
                equalization_criterion=equalization_criterion,
                pruning_granularity=pruning_granularity,
                pth=pth,
                min_num_filters=min_num_filters,
                excluded_layers=excluded_layers)

    s3_output_subdir = 'od_' + model + '_prune_' + timestamp
    #Upload the pruned weights into the object store
    s3.Bucket(bucket).upload_file(os.path.join(WORKSPACE_DIR, pruned_model_filename), os.path.join(s3_output_subdir, pruned_model_filename))
    #Upload the spec file into the object store
    s3.Bucket(bucket).upload_file(os.path.join(SPEC_DIR, spec_filename), os.path.join(s3_output_subdir, spec_filename))

    if model == 'frcnn':
        s3.meta.client.upload_file(os.path.join(WORKSPACE_DIR, zipped_tfrecords), bucket, os.path.join(s3_output_subdir, zipped_tfrecords))

    print('Pruned model has been uploaded to bucket: ', bucket, 'subdir: ', s3_output_subdir)

