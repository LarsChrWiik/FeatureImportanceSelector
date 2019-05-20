
from FeatureImportanceSelector import FeatureImportanceSelector


fis = FeatureImportanceSelector(
    filename='ClassificationData.csv',
    csv_seperator=',',
    output_filename='ClassificationDataNew.csv',
    output_csv_seperator=';',
    drop_columns=[],
    target='TheLabel'
)

score = fis.calculate_accuracy_score()
print(score)

fis.plot_feature_importances()

fis.feature_selection(num_features=2)
