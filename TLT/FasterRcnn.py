import subprocess
import os
from google.protobuf import text_format

class FasterRcnn:

    def __init__(self):
        self.key = 'nvidia_tlt'

    def train(self,
            path_to_dataset_config_file,
            path_to_experiment_spec_file,
            path_to_data_dir,
            path_to_pretrained_weights,
            path_to_output_dir,
            no_of_gpus=1,
            gpu_indices=[0],
            num_processes=-1,
            use_amp=False,
            path_to_log_file=None,
            training_type='new_training',
            hyper_parameters = []):
        
        assert training_type in ['new_training', 'continue_training', 'retrain_pruned']
        command_type = 'training'
        if training_type == 'new_training' or training_type == 'continue_training':
            if os.path.exists(os.path.join(path_to_data_dir, 'tfrecords')) == False:
                print('Generating tfrecords...')
                self.generate_tfrecords(path_to_dataset_config_file, path_to_data_dir, command_type)

        from iva.faster_rcnn.proto import experiment_pb2
        with open(path_to_experiment_spec_file) as f:
            experiment_config = experiment_pb2.Experiment()
            experiment_config = text_format.Merge(f.read(), experiment_config)

        arch = experiment_config.model_config.arch.replace(':', '_')
        experiment_config.training_config.output_model = os.path.join(path_to_output_dir, 'frcnn_' + arch + '.tlt')
        experiment_config.dataset_config.data_sources[0].tfrecords_path = os.path.join(path_to_data_dir, 'tfrecords/tfrecords*')
        experiment_config.dataset_config.data_sources[0].image_directory_path = path_to_data_dir # os.path.join(path_to_data_dir, 'images')
        
        if training_type == 'new_training':
            experiment_config.training_config.pretrained_weights = path_to_pretrained_weights
        elif training_type == 'continue_training':
            experiment_config.training_config.pretrained_weights = ''
            experiment_config.training_config.resume_from_model = path_to_pretrained_weights
        elif training_type == 'retrain_pruned':
            experiment_config.training_config.pretrained_weights = ''
            experiment_config.training_config.retrain_pruned_model = path_to_pretrained_weights
        
        experiment_config.inference_config.images_dir = os.path.join(path_to_data_dir, 'images')

        #if hyper_parameters:

        with open(path_to_experiment_spec_file, 'w') as f:
            f.write(text_format.MessageToString(experiment_config))

        train_command = self.generate_train_command(path_to_experiment_spec_file,
                                                    no_of_gpus,
                                                    gpu_indices,
                                                    num_processes,
                                                    use_amp,
                                                    path_to_log_file)

        # start the training process
        print('Starting training...')
        print('Train command: ', train_command)
        subprocess.run(train_command)
    
    def inference(self, 
            path_to_trained_model,
            input_images_dir,
            path_to_data_dir,
            output_images_dir,
            output_labels_dir,
            path_to_experiment_spec_file,
            draw_conf_thres=0.3,
            gpu_index=0):
        from iva.faster_rcnn.proto import experiment_pb2
        with open(path_to_experiment_spec_file) as f:
            experiment_config = experiment_pb2.Experiment()
            experiment_config = text_format.Merge(f.read(), experiment_config)

        experiment_config.inference_config.images_dir = input_images_dir
        experiment_config.inference_config.model = path_to_trained_model
        experiment_config.inference_config.detection_image_output_dir = output_images_dir
        experiment_config.inference_config.labels_dump_dir = output_labels_dir
        #Specify the confidence threshold in configuration file
        experiment_config.inference_config.object_confidence_thres = draw_conf_thres

        #Specifying the path of tfrecords and images used for training in configuration file
        experiment_config.dataset_config.data_sources[0].tfrecords_path = os.path.join(path_to_data_dir, 'tfrecords/tfrecords*')
        experiment_config.dataset_config.data_sources[0].image_directory_path = path_to_data_dir

        #Specify the pretrained weights to be '' (none)
        experiment_config.training_config.pretrained_weights = ''
        experiment_config.training_config.retrain_pruned_model = ''

        with open(path_to_experiment_spec_file, 'w') as f:
            f.write(text_format.MessageToString(experiment_config))

        inference_command = self.generate_inference_command(path_to_experiment_spec_file, gpu_index)

        print('Starting inference...')
        print('Inference command: ', inference_command)
        subprocess.run(inference_command)

    def prune(self, 
            path_to_trained_frcnn_model, 
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):
        prune_command = self.generate_prune_command(path_to_trained_frcnn_model, 
                                                    path_to_output_file,
                                                    normalizer,
                                                    equalization_criterion,
                                                    pruning_granularity,
                                                    pth,
                                                    min_num_filters,
                                                    excluded_layers)

        print('Starting pruning...')
        print('Prune command: ', prune_command)
        subprocess.run(prune_command)

    def evaluate(self,
            path_to_data_dir,
            path_to_experiment_spec_file,
            path_to_trained_frcnn_model,
            path_to_dataset_config_file,
            evaluation_set = 'validation'):

            assert evaluation_set in ['test', 'validation']
            #Editing the spec file
            from iva.faster_rcnn.proto import experiment_pb2
            with open(path_to_experiment_spec_file) as f:
                experiment_config = experiment_pb2.Experiment()
                experiment_config = text_format.Merge(f.read(), experiment_config)

            #Specifying the path of tfrecords and images in configuration file
            experiment_config.dataset_config.data_sources[0].tfrecords_path = os.path.join(path_to_data_dir, 'tfrecords/tfrecords*')
            experiment_config.dataset_config.data_sources[0].image_directory_path = path_to_data_dir
            #Specifying the path of trained weights in configuration file
            experiment_config.evaluation_config.model = path_to_trained_frcnn_model
            #Specify the pretrained weights to be '' (none)
            experiment_config.training_config.pretrained_weights = ''
            #Specify the path of images for inference in config files
            experiment_config.inference_config.images_dir = os.path.join(path_to_data_dir, 'images')

            if evaluation_set == 'test':
                command_type = 'evaluating'
                print('Generating tfrecords...')
                self.generate_tfrecords(path_to_dataset_config_file, os.path.join(path_to_data_dir,'test'), command_type)
                #Editing the spec file
                experiment_config.dataset_config.validation_data_source.tfrecords_path = os.path.join(path_to_data_dir, 'test/tfrecords/tfrecords*')
                experiment_config.dataset_config.validation_data_source.image_directory_path = os.path.join(path_to_data_dir, 'test')
                #experiment_config.dataset_config.image_extension = 'jpg'
                #Have to remove the validation_fold: 0 line in order to use the whole test data to evaluate the model
                with open(path_to_experiment_spec_file, 'r') as f:
                    lines = f.readlines()
                f.close()

                with open(path_to_experiment_spec_file, 'w') as fs:
                    for line in lines:
                        if line.strip("\n") != 'validation_fold: 0':
                            fs.write(line)
                fs.close()

            with open(path_to_experiment_spec_file, 'w') as spec_f:
                spec_f.write(text_format.MessageToString(experiment_config))

        
            evaluate_command = self.generate_evaluate_command(path_to_experiment_spec_file)


            print('Starting evaluation...')
            print('Evaluate command: ', evaluate_command)
            subprocess.run(evaluate_command)

    def export(self,
            path_to_trained_frcnn_model,
            path_to_experiment_spec_file,
            path_to_data_dir,
            path_to_output_file,
            data_type='fp32'):

            from iva.faster_rcnn.proto import experiment_pb2
            with open(path_to_experiment_spec_file) as f:
                experiment_config = experiment_pb2.Experiment()
                experiment_config = text_format.Merge(f.read(), experiment_config)
            
            #Specify the path of the tfrecords and images in config file
            experiment_config.dataset_config.data_sources[0].tfrecords_path = os.path.join(path_to_data_dir, 'tfrecords/tfrecords*')
            experiment_config.dataset_config.data_sources[0].image_directory_path = path_to_data_dir 

            #Specify the the path of the pretrained weights and retrain_pruned_model in the config files to be '' (none)
            experiment_config.training_config.pretrained_weights = ''
            experiment_config.training_config.retrain_pruned_model = ''
            #Specify the path of images for inference in config files
            experiment_config.inference_config.images_dir = os.path.join(path_to_data_dir, 'images')
            
            with open(path_to_experiment_spec_file, 'w') as f:
                f.write(text_format.MessageToString(experiment_config))

            export_command = self.generate_export_command(path_to_trained_frcnn_model, path_to_experiment_spec_file, path_to_output_file, data_type)

            print('Starting export...')
            print('Export command: ', export_command)
            subprocess.run(export_command)

    def generate_tfrecords(self, 
            path_to_dataset_config_file, 
            path_to_data_dir,
            command_type = 'training'):
        from iva.detectnet_v2.proto import dataset_export_config_pb2
        with open(path_to_dataset_config_file) as f:
            dataset_config = dataset_export_config_pb2.DatasetExportConfig()
            dataset_config = text_format.Merge(f.read(), dataset_config)

        if command_type == 'training':
            print('Generating tfrecords for train data...')
            #To be the same as the image_directory_path specified in the spec file
            dataset_config.image_directory_path = path_to_data_dir # os.path.join(path_to_data_dir, 'training')
            dataset_config.kitti_config.root_directory_path = path_to_data_dir # os.path.join(path_to_data_dir, 'training')
            dataset_config.kitti_config.image_dir_name = 'images'
            dataset_config.kitti_config.label_dir_name = 'labels'
        elif command_type == 'evaluating':
            print('Generating tfrecords for test data...')
            dataset_config.image_directory_path = os.path.join(path_to_data_dir) # os.path.join(path_to_data_dir, 'training')
            dataset_config.kitti_config.root_directory_path = os.path.join(path_to_data_dir) # os.path.join(path_to_data_dir, 'training')
            dataset_config.kitti_config.image_dir_name = 'images'
            dataset_config.kitti_config.label_dir_name = 'labels'
            #dataset_config.kitti_config.image_extension = '.jpg'

        with open(path_to_dataset_config_file, 'w') as f:
            f.write(text_format.MessageToString(dataset_config))

        dataset_convert_command = self.generate_dataset_convert_command(path_to_dataset_config_file, 
                                                                        os.path.join(path_to_data_dir, 'tfrecords/tfrecords'))

        subprocess.run(dataset_convert_command)

    def generate_train_command(self,
            path_to_experiment_spec_file,
            no_of_gpus=1,
            gpu_indices=[0],
            num_processes=-1,
            use_amp=False,
            path_to_log_file=None):

        # generate the faster_rcnn train command
        train_command = [
            'faster_rcnn', 'train',
            '-e', path_to_experiment_spec_file,
            '-k', self.key,
            '--gpus', str(no_of_gpus),
            '--gpu_index'
        ]
        for gpu_index in gpu_indices:
            train_command.append(str(gpu_index))
        if use_amp:
            train_command.extend(['--use_amp'])
        if path_to_log_file:
            train_command.extend(['--log_file', path_to_log_file])

        return train_command

    def generate_dataset_convert_command(self,
            path_to_dataset_config_file,
            path_to_output_tfrecords,
            gpu_index=0):

        # generate the faster_rcnn dataset_convert command
        dataset_convert_command = [
            'faster_rcnn', 'dataset_convert',
            '-d', path_to_dataset_config_file,
            '-o', path_to_output_tfrecords,
            '--gpu_index', str(gpu_index)
        ]

        return dataset_convert_command

    def generate_inference_command(self,
            path_to_experiment_spec_file,
            gpu_index):

        #generate the faster_rcnn inference command
        inference_command = [
            'faster_rcnn', 'inference',
            '-e', path_to_experiment_spec_file,
            '-k', self.key,
            '--gpu_index', str(gpu_index)
        ]

        return inference_command

    def generate_evaluate_command(self,
            path_to_experiment_spec_file,
            gpu_index=0):

        # generate the faster_rcnn evaluate command
        evaluate_command = [
            'faster_rcnn', 'evaluate',
            '-e', path_to_experiment_spec_file,
            '-k', self.key,
            '--gpu_index', str(gpu_index)
        ]

        return evaluate_command

    def generate_prune_command(self,
            path_to_trained_frcnn_model,
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):

        # generate the faster_rcnn prune command
        prune_command = [
            'faster_rcnn', 'prune',
            '-m', path_to_trained_frcnn_model,
            '-o', path_to_output_file,
            '-k', self.key,
            '--gpu_index', '0',
            '--normalizer', normalizer,
            '--equalization_criterion', equalization_criterion,
            '--pruning_granularity', str(pruning_granularity),
            '-pth', str(pth),
            '--min_num_filters', str(min_num_filters),
        ]
        if excluded_layers:
            prune_command.append('--excluded_layers')
            for layer in excluded_layers:
                prune_command.append(layer)

        return prune_command

    def generate_export_command(self,
            path_to_trained_frcnn_model,
            path_to_experiment_spec_file,
            path_to_output_file,
            data_type):

        export_command = [
            'faster_rcnn', 'export',
            '-m', path_to_trained_frcnn_model,
            '-k', self.key,
            '-e', path_to_experiment_spec_file,
            '-o', path_to_output_file,
            '--data_type', data_type
        ]

        return export_command