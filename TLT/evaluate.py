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

def create_new_experiment_workspace(model, timestamp):
    # create workspace dir
    workspace_dir = os.path.join('/workspace', 
                                model + '_evaluate_' + timestamp)
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
    print(AWS_ACCESS_KEY , AWS_SECRET_ACCESS)

    # clearml stuff
    task = Task.init(project_name="DVD-AI", task_name="Frcnn Resnet 18 Drone 6 Epoch Evaluate Test Set", task_type='testing')
    print ("+")
    task.set_base_docker(f"harbor.dsta.ai/nvidia/tlt_trt7.2.1:v1 --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}")
    print ("++")
    task.execute_remotely(queue_name="queue-2xV100-128ram", exit_process=True)
    print ("+++")

    #Specify the model used for evaluation
    models = ['yolov4', 'ssd', 'frcnn']
    model = 'frcnn'

    # check validity of selected model
    assert model in models 

    #Specify the evaluation set used to evaluate model performance
    evaluation_sets = ['test', 'validation']
    evaluation_set = 'test'

    #check the validity of selected evaluation set
    assert evaluation_set in evaluation_sets

    # specify the bucket and the files in the object store to be fetched
    zip_data_filename = 'drone_data_unsplit.zip'
    spec_subdir = 'od_frcnn_train_2021-10-05-14:42:46/'
    spec_filename = 'frcnn_drone_train_resnet18_kitti.txt'
    trained_model_subdir = 'od_yolov4_prune_2021-11-24-11:02:57/'
    trained_model_filename = 'frcnn_resnet_18.epoch54.tlt'

    if model == 'frcnn':
        tfrecords_subdir = 'od_frcnn_train_2021-10-05-14:42:46/'
        zipped_tfrecords = 'tfrecords.zip'
        dataset_config_filename_subdir = 'specs/'
        dataset_config_filename = 'frcnn_drone_config_resnet18_tfrecords.txt'

    if evaluation_set == 'test':
        zip_test_data_filename = 'test2.zip'

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
    print('downloading Data, Spec and Trained model...')
    s3.Bucket(bucket).download_file(os.path.join(trained_model_subdir, trained_model_filename), 
                                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename))
    s3.Bucket(bucket).download_file(os.path.join(spec_subdir, spec_filename),
                                    os.path.join(SPEC_DIR, spec_filename))
    s3.Bucket(bucket).download_file(zip_data_filename, os.path.join(WORKSPACE_DIR, zip_data_filename))

    # unzip the data file
    print('unzipping...')
    if zip_data_filename.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zip_data_filename),'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])
    elif zip_data_filename.endswith('.tar.gz'):
        import tarfile
        with tarfile.open(os.path.join(WORKSPACE_DIR, zip_data_filename),'r') as tar_ref:
            tar_ref.extractall(DATA_DIR)
            DATA_DIR = os.path.join(DATA_DIR, list(os.listdir(DATA_DIR))[0])

    if model == 'frcnn':
        print('downloading the zipped tfrecords...')
        s3.Bucket(bucket).download_file(os.path.join(tfrecords_subdir, zipped_tfrecords), os.path.join(WORKSPACE_DIR, zipped_tfrecords))

        print('unzipping the zipped tfrecords')
        if zipped_tfrecords.endswith('.zip'):
            print('Unzipping tfrecords...')
            import zipfile
            with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zipped_tfrecords), 'r') as zipped_ref:
                zipped_ref.extractall(os.path.join(DATA_DIR, 'tfrecords'))
        elif zipped_tfrecords.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(os.path.join(WORKSPACE_DIR,zipped_tfrecords), 'r') as tarred_ref:
                tarred_ref.extractall(os.path.join(DATA_DIR, 'tfrecords'))

    if evaluation_set == 'test':
        print('downloading the zipped test data...')
        s3.Bucket(bucket).download_file(zip_test_data_filename , os.path.join(WORKSPACE_DIR,  zip_test_data_filename))
        
        print('Unzipping test data...')
        if zip_test_data_filename.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(os.path.join(WORKSPACE_DIR, zip_test_data_filename), 'r') as zipped_test:
                zipped_test.extractall(os.path.join(DATA_DIR, 'test'))
        elif zip_test_data_filename.endswith('.zip'):
            import tarfile
            with tarfile.open(os.path.join(WORKSPACE_DIR, zip_test_data_filename), 'r') as tarred_ref:
                tarred_ref.extractall(os.path.join(DATA_DIR, 'test'))


    #Evaluation command for the different models
    if model == 'yolov4':
        #Start the evaluation
        yolov4 = Yolov4()
        yolov4.evaluate(DATA_DIR, 
                        os.path.join(SPEC_DIR, spec_filename), 
                        os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                        evaluation_set = evaluation_set)

    elif model == 'ssd':
        #Start the evaluation
        ssd = Ssd()
        ssd.evaluate(DATA_DIR,
                    os.path.join(SPEC_DIR, spec_filename),
                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                    evaluation_set = evaluation_set)
            
    elif model == 'frcnn':
        s3.Bucket(bucket).download_file(os.path.join(dataset_config_filename_subdir, dataset_config_filename), 
                                        os.path.join(SPEC_DIR, dataset_config_filename))
        frcnn = FasterRcnn()
        #Start the evaluation
        frcnn.evaluate(DATA_DIR,
                    os.path.join(SPEC_DIR, spec_filename),
                    os.path.join(TRAINED_MODEL_DIR, trained_model_filename),
                    os.path.join(SPEC_DIR, dataset_config_filename),
                    evaluation_set = evaluation_set)
