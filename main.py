import gym
import random
from time import sleep
import numpy as np
# from IPython.display import clear_output # caso você não usa o Jupyer ou o Colab, pode comentar ou apagar essa linha de código

env = gym.make('Taxi-v3').env
env.reset()
env.render()

# 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup and 5 = dropoff
print(env.action_space)
print(env.observation_space)

len(env.P)

print(env.P[484])
print(env.P[384])

q_table = np.zeros([env.observation_space.n, env.action_space.n])

print(q_table.shape)
print(q_table)

print(np.argmax(np.array([3, 5])))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(100000):
    estado = env.reset()
    
    penalidades, recompensa = 0, 0
    done = False
    
    while not done:
        # exploracao
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()
            # exploitation
        else:
            acao = np.argmax(q_table[estado])
        
        proximo_estado, recompensa, done, info = env.step(acao)
        
        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])
        
        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo
        
        if recompensa == -10:
            penalidades += 1
            
        estado = proximo_estado
    
    if i % 100 == 0:
        # clear_output(wait=True)
        print(f'Episódio {i}')
        
print('Treinamento concluído')

q_table[424]

env.reset()
env.render()

# azul, ponto de partida
# roxo, destino final

env.encode(2, 4, 0, 3)

q_table[283]

total_penalidades = 0
episodios = 50
frames = []

for _ in range(episodios):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False
    
    while not done:
        acao = np.argmax(q_table[estado])
        estado, recompensa, done, info = env.step(acao)
        
        if recompensa == -10:
            penalidades += 1
            
        frames.append({'frame': env.render(mode='ansi'),
                      'state': estado,
                       'action': acao,
                       'reward': recompensa
                      })
        
        total_penalidades += penalidades
        
print(f'Episódios {episodios}')
print(f'Penalidades {total_penalidades}')

frames[0]

for frame in frames:
    #clear_output(wait=True)
    
    print(frame['frame'])
    print(f"Estado {frame['state']}")
    print(f"Ação {frame['action']}")
    print(f"Recompensa {frame['reward']}")
    sleep(0.5)