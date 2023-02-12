def get_params():
    """get parameters from user"""
    list_params = [[]]
    list_params[0].append(input('Gender (M/F): '))
    list_params[0].append(int(input('Age: ')))
    list_params[0].append(input('Smoking (Y/N): '))
    list_params[0].append(input('Yellow Fingers (Y/N): '))
    list_params[0].append(input('Anxiety (Y/N): '))
    list_params[0].append(input('Peer Pressure (Y/N): '))
    list_params[0].append(input('Chronic Disease (Y/N): '))
    list_params[0].append(input('Fatigue (Y/N): '))
    list_params[0].append(input('Allergy (Y/N): '))
    list_params[0].append(input('Wheezing (Y/N): '))
    list_params[0].append(input('Alcohol Consuming (Y/N): '))
    list_params[0].append(input('Coughing (Y/N): '))
    list_params[0].append(input('Shortness of Breath (Y/N): '))
    list_params[0].append(input('Swallowing Difficulty (Y/N): '))
    list_params[0].append(input('Chest Pain (Y/N): '))
    
    for i in range(len(list_params[0])):
        if list_params[0][i] == 'M' or list_params[0][i] == 'Y':
            list_params[0][i] = 1
        elif list_params[0][i] == 'F' or list_params[0][i] == 'N':
            list_params[0][i] = 0

    return list_params

def predict_prob(list_params, scaler, pca, model):
    """predict probability of lung cancer development"""
    if scaler != None:
        list_params = scaler.transform(list_params)
        
    if pca != None:
        list_params = pca.transform(list_params)
    
    y_pred = model.predict_proba(list_params)
    
    return y_pred[0][1]
    
def insurance_rec(prob):
    """provide insurance recommendations"""
    insur_dict = {
        'Basic': {
            'Health - Hospitalized Treatment Expense': '100-200K',
            'Health - Hospitalized Surgery Expense': '50-100K',
            'Health - Non-hospitalized Surgery Expense': '50-100K',
            'Health - Daily Hospital Ward Cost': '2K',
            'Cancer - Surgery Expense': None,
            'Cancer - Daily Hospital Ward Cost': None,
            'Cancer - Severe Cancer Payment': None,
            'Critical Illness - Severe Cancer Payment': None
        },
        'Standard': {
            'Health - Hospitalized Treatment Expense': '200-300K',
            'Health - Hospitalized Surgery Expense': '100-200K',
            'Health - Non-hospitalized Surgery Expense': '100-200K',
            'Health - Daily Hospital Ward Cost': '3K',
            'Cancer - Surgery Expense': '50-100K',
            'Cancer - Daily Hospital Ward Cost': '2K',
            'Cancer - Severe Cancer Payment': None,
            'Critical Illness - Severe Cancer Payment': None
        },
        'Premium': {
            'Health - Hospitalized Treatment Expense': '300-400K',
            'Health - Hospitalized Surgery Expense': '200-250K',
            'Health - Non-hospitalized Surgery Expense': '200-250K',
            'Health - Daily Hospital Ward Cost': '4K',
            'Cancer - Surgery Expense': '100-200K',
            'Cancer - Daily Hospital Ward Cost': '3K',
            'Cancer - Severe Cancer Payment': '2M',
            'Critical Illness - Severe Cancer Payment': '1.5M'
        }
    }
    
    if prob > 26 and prob <= 50:
        insur = insur_dict['Basic']
    elif prob > 50 and prob <= 75:
        insur = insur_dict['Standard']
    elif prob > 75:
        insur = insur_dict['Premium']
    else:
        insur = None
    
    return insur

def print_rec(insur):
    """print out insurance recommendations"""
    print('-' * 15)
    print('Insurance Recommendations:')
    
    if insur == None:
        print('None')
    else:
        for key, value in insur.items():
            if value != None:
                print(key + ':', value)
    
    return None

def insurance_appl(scaler, pca, model):
    """apply probability to insurance application"""
    list_params = get_params()
    
    prob = round(predict_prob(list_params, scaler, pca, model) * 100, 2)
#    print('prob:', prob, '%')
    
    insur = insurance_rec(prob)

    print_rec(insur)