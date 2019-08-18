import numpy as np



### Constants

NUM_VARIABLES = 17
INPUT_DIM = 65
GLASCOW_EYE_MAX = 4
GLASCOW_MOTOR_MAX = 6
GLASCOW_VERBAL_MAX = 5
GLASCOW_TOTAL_MAX = 15



### Formatting Data

def format_data_for_patient(patient_data):
	assert len(patient_data.shape) == 2, 'Expected each array to have dimension 2, got {}'.format(len(patient_data.shape))
	assert patient_data.shape[-1] == NUM_VARIABLES, 'Expected number of variables to be {}, got {}'.format(NUM_VARIABLES, patient_data.shape[-1])

	patient_data_new = np.zeros(len(patient_data), INPUT_DIM)

	# Capillary refill rate
	patient_data_new[:, 17] = (patient_data[:, 0] == 0).astype(float)
	patient_data_new[:, 18] = 1 - patient_data_new[:, 0]

	# Diastolic blood pressure
	patient_data_new[:, 19] = patient_data[:, 1]

	# Fraction inspired oxygen
	patient_data_new[:, 20] = patient_data[:, 2]

	# Glascow coma scale eye opening 
	for i in range(GLASCOW_EYE_MAX + 1):
		patient_data_new[:, 21 + i] = (patient_data[:, 3] == i).astype(float)

	# Glascow coma scale motor response
	for i in range(GLASCOW_MOTOR_MAX + 1):
		patient_data_new[:, 26 + i] = (patient_data[:, 4] == i).astype(float)

	# Glascow coma scale total
	for i in range(GLASCOW_TOTAL_MAX + 1):
		patient_data_new[:, 33 + i] = (patient_data[:, 5] == i).astype(float)

	# Glascow coma scale verbal response
	for i in range(GLASCOW_VERBAL_MAX + 1):
		patient_data_new[:, 49 + i] = (patient_data[:, 6] == i).astype(float)

	# Glucose, heart rate, height, mean blood pressure, O2 saturation, 
	# respiratory rate, systolic blood pressure, temperature, weight, pH
	patient_data_new[:, 56:INPUT_DIM + 1] = (patient_data[:, 7:NUM_VARIABLES + 1] == i).astype(float)

	return patient_data_new


def format_data(data):
	formatted = [format_data_for_patient(patient_data) for patient_data in data]
	return pad


def pad(data):
	data = format_data(data)
	check_data(data)

	X = np.zeros((len(data), SEQ_LENGTH, INPUT_DIM))
	X[:, 0] = ADMIT_EMBEDDING

	for i, array in enumerate(data):
		X[i][1 : len(array[:SEQ_LENGTH - 1]) + 1] = array[:SEQ_LENGTH - 1]

	return X


