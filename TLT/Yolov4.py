import subprocess
import sys
import os
import re
from google.protobuf import text_format

class Yolov4: 
 
    def __init__(self):
        self.key = 'nvidia_tlt'

    def train(self, 
            path_to_experiment_spec_file, 
            path_to_data_dir, 
            path_to_output_dir, 
            path_to_pretrained_model,
            no_of_gpus=1,
            gpu_indices=[0],
            use_amp=False,
            training_type='new_training'):
        assert training_type in ['new_training', 'continue_training', 'retrain_pruned']

        # update train and val data source and pretrained model path
        from iva.yolo_v4.proto import experiment_pb2
        with open(path_to_experiment_spec_file) as f:
            experiment_config = experiment_pb2.Experiment()
            experiment_config = text_format.Merge(f.read(), experiment_config)

        experiment_config.dataset_config.data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'training/labels')
        experiment_config.dataset_config.data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'training/images')

        experiment_config.dataset_config.validation_data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'val/labels')
        experiment_config.dataset_config.validation_data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'val/images')

        output_width = experiment_config.augmentation_config.output_width
        output_height = experiment_config.augmentation_config.output_height

        if training_type == 'retrain_pruned':
            experiment_config.training_config.pretrain_model_path = ''
            experiment_config.training_config.pruned_model_path = path_to_pretrained_model
        else:
            # generate the anchor shapes
            anchor_shapes = self.generate_anchor_shapes(os.path.join(path_to_data_dir, 'training/labels'), 
                                                        os.path.join(path_to_data_dir, 'training/images'), 
                                                        output_height, 
                                                        output_width)

            # add the anchor shapes into the train config file
            print('Adding the anchor shapes into the yolov4 config file...')
            small_anchor_shapes = 'small_anchor_shape: "[{}, {}, {}]"'.format(anchor_shapes[0], anchor_shapes[1], anchor_shapes[2])
            mid_anchor_shapes = 'mid_anchor_shape: "[{}, {}, {}]"'.format(anchor_shapes[3], anchor_shapes[4], anchor_shapes[5])
            big_anchor_shapes = 'big_anchor_shape: "[{}, {}, {}]"'.format(anchor_shapes[6], anchor_shapes[7], anchor_shapes[8])
            with open(path_to_experiment_spec_file) as f:
                s = f.read()
                s = re.sub('small_anchor_shape.*\"$', small_anchor_shapes, s, flags=re.MULTILINE)
                s = re.sub('mid_anchor_shape.*\"$', mid_anchor_shapes, s, flags=re.MULTILINE)
                s = re.sub('big_anchor_shape.*\"$', big_anchor_shapes, s, flags=re.MULTILINE)

            with open(path_to_experiment_spec_file, 'w') as f:
                f.write(s)
            print('done')

            if training_type == 'new_training':
                experiment_config.training_config.pretrain_model_path = path_to_pretrained_model
            elif training_type == 'continue_training':
                experiment_config.training_config.pretrain_model_path = ''
                experiment_config.training_config.resume_model_path = path_to_pretrained_model

        with open(path_to_experiment_spec_file, 'w') as f:
            f.write(text_format.MessageToString(experiment_config))

        train_command = self.generate_train_command(path_to_experiment_spec_file,
                                                    path_to_output_dir,
                                                    no_of_gpus=no_of_gpus,
                                                    gpu_indices=gpu_indices)

        # start the training process
        print('Starting training...')
        print('Train command: ', train_command)
        subprocess.run(train_command)

    def evaluate(self, 
            path_to_data_dir,
            path_to_experiment_spec_file, 
            path_to_model_file,
            evaluation_set = 'validation'):

        assert evaluation_set in ['test', 'validation']
        # update train and val data source and pretrained model path
        from iva.yolo_v4.proto import experiment_pb2
        with open(path_to_experiment_spec_file) as f:
            experiment_config = experiment_pb2.Experiment()
            experiment_config = text_format.Merge(f.read(), experiment_config)

        experiment_config.dataset_config.data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'training/labels')
        experiment_config.dataset_config.data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'training/images')
        if evaluation_set == 'validation':
            experiment_config.dataset_config.validation_data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'val/labels')
            experiment_config.dataset_config.validation_data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'val/images')
        elif evaluation_set == 'test':
            if os.path.exists(os.path.join(path_to_data_dir, 'test')) == False:
                raise AssertionError('Test Data does not exists')
            experiment_config.dataset_config.validation_data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'test/labels')
            experiment_config.dataset_config.validation_data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'test/images')

        with open(path_to_experiment_spec_file, 'w') as f:
            f.write(text_format.MessageToString(experiment_config))

        evaluate_command = self.generate_evaluate_command(path_to_experiment_spec_file,
                                                        path_to_model_file)

        print('Starting evaluation on ' + evaluation_set + ' set...')
        print('Evaluate command: ', evaluate_command)
        subprocess.run(evaluate_command)

    def prune(self, 
            path_to_experiment_spec_file,
            path_to_trained_yolov4_model,
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):
        prune_command = self.generate_prune_command(path_to_experiment_spec_file,
                                                    path_to_trained_yolov4_model, 
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
    
    def inference(self, 
            path_to_trained_model,
            input_images_dir,
            output_images_dir,
            output_labels_dir,
            path_to_experiment_spec_file,
            draw_conf_thres =0.3,
            gpu_index =0):
        
        inference_command = self.generate_inference_command(path_to_trained_model,
                                                            input_images_dir,
                                                            output_images_dir,
                                                            output_labels_dir,
                                                            path_to_experiment_spec_file,
                                                            draw_conf_thres,
                                                            gpu_index)
        print('Starting inference...')
        print('Inference command: ', inference_command)
        subprocess.run(inference_command)

    def export(self,
            path_to_trained_model,
            path_to_experiment_spec_file,
            path_to_output_file,
            data_type='fp32'):

        export_command = self.generate_export_command(path_to_trained_model, path_to_experiment_spec_file, path_to_output_file, data_type)

        print('Starting export...')
        print('Export command: ', export_command)
        subprocess.run(export_command)

    def generate_anchor_shapes(self,
            path_to_label_folders,
            path_to_training_image_folders,
            input_width,
            input_height,
            num_shape_clusters=9,
            max_steps=10000,
            min_x=0,
            min_y=0):

        kmeans_command = self.generate_kmeans_command(path_to_label_folders,
                                                    path_to_training_image_folders,
                                                    input_width,
                                                    input_height,
                                                    num_shape_clusters,
                                                    max_steps,
                                                    min_x,
                                                    min_y)
        print(kmeans_command)

        print('Generating the anchor shapes...')
        result = subprocess.run(kmeans_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print('Error in running the yolo_v4 kmeans command. Check your parameters!')
            sys.exit() 

        output_string = result.stdout.decode('utf-8')
        anchor_shapes = output_string.strip().split('\n')[-9:]
        print('Generated the following anchor shapes:')
        print(anchor_shapes)

        return anchor_shapes

    def generate_kmeans_command(self, 
            path_to_label_folders,
            path_to_training_image_folders,
            input_width,
            input_height,
            num_shape_clusters=9,
            max_steps=10000,
            min_x=0,
            min_y=0):

        # generate the yolo_v4 kmeans command
        kmeans_command = [
            'yolo_v4', 'kmeans', 
            '-l', path_to_label_folders,
            '-i', path_to_training_image_folders,
            '-n', str(num_shape_clusters),
            '-x', str(input_width),
            '-y', str(input_height),
            '--max_steps', str(max_steps),
            '--min_x', str(min_x),
            '--min_y', str(min_y)
        ]

        return kmeans_command

    def generate_train_command(self,
            path_to_experiment_spec_file,
            path_to_output_dir,
            no_of_gpus=1,
            gpu_indices=[0],
            use_amp=False):

        # generate the yolo_v4 train command
        train_command = [
            'yolo_v4', 'train',
            '-r', path_to_output_dir,
            '-e', path_to_experiment_spec_file,
            '-k', self.key,
            '--gpus', str(no_of_gpus),
            '--gpu_index',
            # '--log_file', path_to_log_file
        ]
        for gpu_index in gpu_indices:
            train_command.append(str(gpu_index))
        if use_amp:
            train_command.extend(['--use_amp'])

        return train_command

    def generate_evaluate_command(self,
            path_to_experiment_spec_file,
            path_to_model_file,
            gpu_index=0):

        # generate the yolo_v4 evaluate command
        evaluate_command = [
            'yolo_v4', 'evaluate',
            '-e', path_to_experiment_spec_file,
            '-m', path_to_model_file,
            '-k', self.key,
            '--gpu_index', str(gpu_index)
        ]

        return evaluate_command

    def generate_prune_command(self,
            path_to_experiment_spec_file,
            path_to_trained_yolov4_model,
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):

        # generate the yolo_v4 prune command
        prune_command = [
            'yolo_v4', 'prune',
            '-e', path_to_experiment_spec_file,
            '-m', path_to_trained_yolov4_model,
            '-o', path_to_output_file,
            '-k', self.key,
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

    def generate_inference_command(self,
            path_to_trained_model,
            input_images_dir,
            output_images_dir,
            out_labels_dir,
            path_to_spec_file,
            draw_conf_thres=0.3,
            gpu_index=0):

        inference_command = [
            'yolo_v4', 'inference',
            '-i', input_images_dir,
            '-o', output_images_dir, 
            '-m', path_to_trained_model,
            '-k', self.key,
            '-e', path_to_spec_file,
            '-t', str(draw_conf_thres),
            '-l', out_labels_dir,
            '--gpu_index', str(gpu_index)
        ]

        return inference_command

    def generate_export_command(self,
            path_to_trained_model,
            path_to_experiment_spec_file,
            path_to_output_file,
            data_type):
        
        export_command = [
            'yolo_v4', 'export',
            '-m', path_to_trained_model,
            '-k', self.key,
            '-e', path_to_experiment_spec_file,
            '-o', path_to_output_file,
            '--data_type', data_type
        ]

        return export_command