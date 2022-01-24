import subprocess
import os
from google.protobuf import text_format

class Ssd:

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
            initial_epoch=0):

        from iva.ssd.proto import experiment_pb2
        with open(path_to_experiment_spec_file) as f:
            experiment_config = experiment_pb2.Experiment()
            experiment_config = text_format.Merge(f.read(), experiment_config)

        experiment_config.dataset_config.data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'training/labels')
        experiment_config.dataset_config.data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'training/images')

        experiment_config.dataset_config.validation_data_sources[0].label_directory_path = os.path.join(path_to_data_dir, 'val/labels')
        experiment_config.dataset_config.validation_data_sources[0].image_directory_path = os.path.join(path_to_data_dir, 'val/images')

        with open(path_to_experiment_spec_file, 'w') as f:
            f.write(text_format.MessageToString(experiment_config))

        train_command = self.generate_train_command(path_to_experiment_spec_file,
                                                    path_to_output_dir,
                                                    path_to_pretrained_model,
                                                    no_of_gpus,
                                                    gpu_indices,
                                                    use_amp,
                                                    initial_epoch)

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

        from iva.ssd.proto import experiment_pb2
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

        print('Starting evaluation...')
        print('Evaluate command: ', evaluate_command)
        subprocess.run(evaluate_command)

    def prune(self,
            path_to_trained_ssd_model,
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):
        prune_command = self.generate_prune_command(path_to_trained_ssd_model, 
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
            path_to_spec_file,
            draw_conf_thres= 0.3,
            gpu_index= 0):
        
        inference_command = self.generate_inference_command(path_to_trained_model,
                                                            input_images_dir,
                                                            output_images_dir,
                                                            output_labels_dir,
                                                            path_to_spec_file,
                                                            draw_conf_thres,
                                                            gpu_index)

        print('Starting inference...')
        print('Inference command: ', inference_command)
        subprocess.run(inference_command)

    def export(self,
            path_to_trained_frcnn_model,
            path_to_experiment_spec_file,
            path_to_output_file,
            data_type='fp32'):

        export_command = self.generate_export_command(path_to_trained_frcnn_model, path_to_experiment_spec_file, path_to_output_file, data_type)

        print('Starting export...')
        print('Export command: ', export_command)
        subprocess.run(export_command)

    def generate_train_command(self,
            path_to_experiment_spec_file,
            path_to_output_dir,
            path_to_pretrained_model,
            no_of_gpus=1,
            gpu_indices=[0],
            use_amp=False,
            training_type='new_training',
            initial_epoch=0):

        # generate the ssd train command
        train_command = [
            'ssd', 'train',
            '-r', path_to_output_dir,
            '-e', path_to_experiment_spec_file,
            '-m', path_to_pretrained_model,
            '-k', self.key,
            '--gpus', str(no_of_gpus),
            '--gpu_index',
        ]
        for gpu_index in gpu_indices:
            train_command.append(str(gpu_index))
        if use_amp:
            train_command.extend(['--use_amp'])
        if initial_epoch:
            train_command.extend(['--initial_epoch', str(initial_epoch)])

        return train_command

    def generate_evaluate_command(self,
            path_to_experiment_spec_file,
            path_to_model_file,
            gpu_index=0):

        # generate the ssd evaluate command
        evaluate_command = [
            'ssd', 'evaluate',
            '-e', path_to_experiment_spec_file,
            '-m', path_to_model_file,
            '-k', self.key,
            '--gpu_index', str(gpu_index)
        ]

        return evaluate_command

    def generate_prune_command(self,
            path_to_trained_ssd_model,
            path_to_output_file,
            normalizer='max',
            equalization_criterion='union',
            pruning_granularity=8,
            pth=0.1,
            min_num_filters=16,
            excluded_layers=[]):

        # generate the ssd prune command
        prune_command = [
            'ssd', 'prune',
            '-m', path_to_trained_ssd_model,
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
            output_labels_dir,
            path_to_spec_file,
            draw_conf_thres=0.3,
            gpu_index= 0):

        inference_command = [
            'ssd', 'inference',
            '-i', input_images_dir,
            '-o', output_images_dir, 
            '-m', path_to_trained_model,
            '-k', self.key,
            '-e', path_to_spec_file,
            '-t', str(draw_conf_thres),
            '-l', output_labels_dir,
            '--gpu_index', str(gpu_index)
        ]

        return inference_command

    def generate_export_command(self,
            path_to_trained_model,
            path_to_experiment_spec_file,
            path_to_output_file,
            data_type):

        export_command = [
            'ssd', 'export',
            '-m', path_to_trained_model,
            '-k', self.key,
            '-e', path_to_experiment_spec_file,
            '-o', path_to_output_file,
            '--data_type', data_type
        ]

        return export_command