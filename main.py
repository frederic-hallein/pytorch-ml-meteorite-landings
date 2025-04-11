""" Main module """

from torch.utils.data import random_split

from lib.data_analysis import get_filtered_csv_data, \
                              plot_data, \
                              convert_categorical_to_numerical, \
                              standardization, \
                              degree_to_radians, \
                              create_dataset
from lib.machine_learning import get_device, create_ml_model, save_model, evaluate
from lib.machine_learning import NeuralNetwork

# paths
PATH       = 'data/archive/meteorite-landings-filtered.csv'
MODEL_PATH = 'models/model.pth'

# model variables
FEATURE_NAME = ['nametype', 'mass', 'fall', 'year', 'reclat', 'reclong']
LABEL_NAME    = 'recclass'

# hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE    = 16
EPOCHS        = 100

def main() -> None:
    """
    Main function.

    :return: None
    """
    # load data ---------------------------------------------
    df = get_filtered_csv_data(PATH)
    plot_data(df)

    # data transformation -----------------------------------
    df, category_mappings = convert_categorical_to_numerical(df,
                            ['nametype', 'recclass', 'fall'])
    df = standardization(df, ['mass', 'year'])
    df = degree_to_radians(df, ['reclat', 'reclong'])

    # machine learning --------------------------------------
    input_size  = len(FEATURE_NAME)
    num_classes = len(df[LABEL_NAME].astype('category').cat.categories)

    dataset = create_dataset(df, FEATURE_NAME, LABEL_NAME)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    device = get_device()
    model = NeuralNetwork(input_size, num_classes).to(device)
    model = create_ml_model(model, train_dataset, test_dataset,
                             LEARNING_RATE, BATCH_SIZE, EPOCHS, device)

    # save model --------------------------------------------
    save_model(model, MODEL_PATH)

    # model evaluation --------------------------------------
    recclass_map = category_mappings[LABEL_NAME]
    evaluate(model, test_dataset, recclass_map, device)

if __name__ == "__main__":
    main()
