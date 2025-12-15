import pyplAI

PLAYER_NUMBER = 2
DEPTHT_MINIMAX = 7
TIME_MCTS = 1.9

# Adapting ConnectX to pyplAI
class ConnectXState:
    def __init__(self, board_data, jugador_actual=1):
        self.board = list(board_data)
        self.jugador_actual = jugador_actual
    
    def obtiene_movimientos(self):
        pass
    def aplica_movimiento(self, movimiento):
        pass
    def es_estado_final(self):
        pass
    def gana_jugador(self, jugador):
        pass
    def heuristica(self):
        pass

minimax_solver = pyplAI.Minimax(
           ConnectXState.aplica_movimiento,
           ConnectXState.obtiene_movimientos, 
           ConnectXState.es_estado_final, 
           ConnectXState.gana_jugador, 
           ConnectXState.heuristica,
           PLAYER_NUMBER, 
           DEPTHT_MINIMAX)

mcts_solver = pyplAI.MCTS(
         ConnectXState.aplica_movimiento,
         ConnectXState.obtiene_movimientos, 
         ConnectXState.es_estado_final, 
         ConnectXState.gana_jugador, 
         PLAYER_NUMBER, 
         TIME_MCTS)


def minmax_agent(observation, configuration):
    current_board = observation.board
    current_player = observation.mark
    
    pyplai_state = ConnectXState(current_board, current_player)
    
    recommended_move = minimax_solver.ejecuta(pyplai_state)
    
    return recommended_move

def mcts_agent(observation, configuration):
    current_board = observation.board
    current_player = observation.mark
    
    pyplai_state = ConnectXState(current_board, current_player)
    
    recommended_move = mcts_solver.ejecuta(pyplai_state)
    
    return recommended_move