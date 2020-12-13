import csv

def to_csv(filename, data, epochs):
    with open(filename, mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'loss', 'accuracy'])

        for i in range(epochs):
            writer.writerow([
                str(i + 1),
                str(data['loss'][i]),
                str(data['accuracy'][i])
            ])

def predictions_to_file(filename, data):
    with open(filename, mode='a') as file:
        file.write(''.join(str(i) for i in data))

def to_file(filename, data):
    with open(filename, mode='a') as file:
        file.write(data)

def get_predictions(scores):
    predictions = []
    index = 0
    _max = -1
    prediction = -1

    for score in scores:
        _max = -1
        prediction = -1
        index = 0
        for value in score:
            if value > _max:
                _max = value
                prediction = index
            
            index += 1
        predictions.append(prediction)

    return predictions