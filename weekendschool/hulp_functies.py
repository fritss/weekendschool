from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata
import flatbuffers

try:
    import matplotlib.pyplott as plt
except ImportError as e:
    pass


def plot_model(model_details, filename=None):

    # Create sub-plots
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_details.history['accuracy'])+1),model_details.history['accuracy'])
    axs[0].plot(range(1,len(model_details.history['val_accuracy'])+1),model_details.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['accuracy'])+1))
    axs[0].legend(['train', 'val'], loc='best')
    axs[0].set_ylim(0,1.1)
    
    # Summarize history for loss
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1))
    axs[1].legend(['train', 'val'], loc='best')
    
    # Save and show the plot
    if filename != None:
        plt.savefig(filename)
    plt.show()



def plot_images(images, labels_true, class_names, labels_pred=None):
    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    cols = 6
    rows = (len(images)+cols-1) // cols
    fig, axes = plt.subplots(rows, cols, figsize = (15,7))

    # Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.1
    else:
        hspace = 0.3
    fig.subplots_adjust(hspace=hspace, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')
            
            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: "+labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Show the plot
    plt.show()
    
def generate_metadata(info):
    # Copies model_file to export_path.
    tf.io.gfile.copy(str(info['model_path']), str(info['export_model_path']), overwrite=False)

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "MobileNet image classifier for objects"
    model_meta.description = (f"Noem het meest aanwezige voorwerp van ({', '.join(info['classes'])})")
    model_meta.version = "v1"
    model_meta.author = info['author']
    model_meta.license = ("Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (f"Input image to be classified. The expected image is {info['size']} x {info['size']},  "
                              "with three channels (red, blue, and green) per pixel. Each value in the "
                              "tensor is a single byte between 0 and 1.")
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (_metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.ImageProperties)

    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [0.0]
    input_normalization.options.std = [255.0]
    input_meta.processUnits = [input_normalization]

    input_stats = _metadata_fb.StatsT()
    input_stats.max = [1]
    input_stats.min = [0]
    input_meta.stats = input_stats

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = f"Probabilities of the {len(info['classes'])} labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats

    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = info['label_fn']
    label_file.description = "Labels for objects that the model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # Populate metadata and label file to the model file.
    populator = _metadata.MetadataPopulator.with_model_file(str(info['export_model_path']))
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([str(info['label_path'])])
    populator.populate()

    displayer = _metadata.MetadataDisplayer.with_model_file(str(info['export_model_path']))
    json_file = displayer.get_metadata_json()
    with open(info['json_fn'], "w") as f:
        f.write(json_file)
