# Kaggle submissions cannot install pyplAI, so the Minimax algorithm is vendored below verbatim from
# pyplAI v1.1.0 (https://github.com/plss12/pyplAI/blob/v1.1.0/pyplAI/Algoritmos.py).
# tests/test_pyplai_sync.py checks that it still matches the installed pyplAI.
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List

#Valor de una victoria en Minimax. Las heurísticas deben devolver valores de menor magnitud
VICTORIA = 1e9


@dataclass
class EvaluacionMovimiento:
    """Evaluación de un movimiento de la raíz tras la búsqueda."""
    movimiento: Any
    #Número de visitas del nodo (0 en Minimax, que no muestrea)
    visitas: int
    #Recompensa media del jugador que mueve en MCTS e ISMCTS (entre 0 y 1) o valor minimax
    valor: float


@dataclass
class ResultadoAnalisis:
    """Resultado de analizar un estado: movimiento elegido y ranking de todos los movimientos."""
    movimiento: Any
    #Movimientos ordenados de mejor a peor según el criterio de elección del algoritmo
    movimientos: List[EvaluacionMovimiento] = field(default_factory=list)
    #Iteraciones de búsqueda (Monte Carlo) o estados evaluados (Minimax)
    iteraciones: int = 0
    nodos: int = 0
    tiempo: float = 0.0


def _comprueba_movimientos(movimientos: list, estado: Any) -> None:
    if len(movimientos) == 0:
        raise ValueError(
            "El estado no es final pero no tiene movimientos posibles. Todo estado no final "
            "debe tener al menos un movimiento (estado: " + repr(estado) + ")")


class Minimax:
    """Minimax con poda alfa-beta (Knuth y Moore, 1975).

    Con más de dos jugadores es paranoico: el resto de jugadores minimiza el valor del jugador
    que busca. profundidadBusqueda es el número de niveles (jugadas) que se exploran.
    """

    def __init__(self, aplicar_movimiento, obtener_movimientos, es_estado_final, gana_jugador, heuristica, numeroJugadores, profundidadBusqueda, estadisticas=False):
        if profundidadBusqueda < 1:
            raise ValueError("profundidadBusqueda debe ser un entero mayor que 0")
        self.aplicar_movimiento = aplicar_movimiento
        self.obtener_movimientos = obtener_movimientos
        self.es_estado_final = es_estado_final
        self.gana_jugador = gana_jugador
        self.heuristica = heuristica
        self.jugadores = [i+1 for i in range(numeroJugadores)]
        self.profundidadBusqueda = profundidadBusqueda
        self.estadisticas = estadisticas

    def evaluate(self,estado,jugadorMax,nivel=0):
        #Las victorias valen más cuanto antes llegan y las derrotas, cuanto más tarde
        if(self.es_estado_final(estado)):
            for jugador in self.jugadores:
                if(self.gana_jugador(estado,jugador)):
                    if(jugador==jugadorMax):
                        return VICTORIA - nivel
                    else:
                        return -VICTORIA + nivel
            return 0
        return self.heuristica(estado,jugadorMax)

    def alfabeta(self, estado, nivel, jugadorMax, alpha, beta, contador):
        """Valor minimax de estado, situado a nivel jugadas de la raíz."""
        contador[0] += 1
        if(self.es_estado_final(estado) or nivel>=self.profundidadBusqueda):
            return self.evaluate(estado,jugadorMax,nivel)
        movs = self.obtener_movimientos(estado)
        _comprueba_movimientos(movs, estado)
        if(estado.jugadorActual==jugadorMax):
            best = -math.inf
            for mov in movs:
                hijo = self.aplicar_movimiento(deepcopy(estado),mov)
                best = max(best, self.alfabeta(hijo, nivel+1, jugadorMax, alpha, beta, contador))
                alpha = max(alpha, best)
                if(beta<=alpha):
                    break
        else:
            best = math.inf
            for mov in movs:
                hijo = self.aplicar_movimiento(deepcopy(estado),mov)
                best = min(best, self.alfabeta(hijo, nivel+1, jugadorMax, alpha, beta, contador))
                beta = min(beta, best)
                if(beta<=alpha):
                    break
        return best

    def _busca(self, estado, completo):
        t0 = time.perf_counter()
        jugadorMax = estado.jugadorActual
        contador = [1]
        evaluaciones = []
        alpha = -math.inf
        for mov in self.obtener_movimientos(estado):
            hijo = self.aplicar_movimiento(deepcopy(estado),mov)
            #Con completo=True cada movimiento se evalúa con ventana completa y su valor es exacto;
            #si no, los movimientos peores que el mejor encontrado solo se acotan (poda en la raíz)
            ventana = -math.inf if completo else alpha
            valor = self.alfabeta(hijo, 1, jugadorMax, ventana, math.inf, contador)
            evaluaciones.append(EvaluacionMovimiento(mov, 0, valor))
            alpha = max(alpha, valor)
        mejor = evaluaciones[0]
        for evaluacion in evaluaciones:
            if(evaluacion.valor>mejor.valor):
                mejor = evaluacion
        ordenadas = sorted(evaluaciones, key=lambda e: e.valor, reverse=True)
        return ResultadoAnalisis(mejor.movimiento, ordenadas, contador[0], contador[0], time.perf_counter()-t0)

    def analiza(self, estado):
        """Valor exacto de cada movimiento de la raíz y movimiento elegido. No imprime nada."""
        if self.es_estado_final(estado) or len(self.obtener_movimientos(estado)) == 0:
            return ResultadoAnalisis(None)
        return self._busca(estado, True)

    def ejecuta(self, estado):
        movs = self.obtener_movimientos(estado)
        if(len(movs)==1):
            return movs[0]
        elif(len(movs)==0 or self.es_estado_final(estado)):
            return None
        resultado = self._busca(estado, False)
        if(self.estadisticas):
            print("\nTiempo de ejecución: ", resultado.tiempo)
            print("Número de estados evaluados: ", resultado.iteraciones)
        return resultado.movimiento


# Adapting ConnectX to pyplAI
class ConnectXState:
    ROWS = 6
    COLUMNS = 7

    def __init__(self, board_data, current_player=1, time_limit_epoch=None):
        self.board = list(board_data)
        self.jugadorActual = current_player
        self.stop_time = time_limit_epoch

    def get_piece(self, row, col):
        return self.board[row * self.COLUMNS + col]
    
    def get_moves(self):
        movs = []

        for col in range(self.COLUMNS):
            if self.board[col] == 0:
                movs.append(col)

        return movs

    def apply_move(self, movement):
        landing_index = -1

        for row in range(self.ROWS - 1, -1, -1):
            index = row * self.COLUMNS + movement

            if self.board[index] == 0:
                landing_index = index
                break
        
        self.board[landing_index] = self.jugadorActual
        self.jugadorActual = 2 if self.jugadorActual == 1 else 1

        return self

    def is_final_state(self):
        num_movs = len(self.get_moves())
        if num_movs == 0:
            return True
        
        if self.wins_player(1) or self.wins_player(2):
            return True
        
        return False

    def wins_player(self, player):
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):

                if self.get_piece(row, col) == player:
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

                    for dr, dc in directions:
                        count = 1
                        
                        for i in range(1, 4):
                            
                            next_row = row + dr * i
                            next_col = col + dc * i

                            if not (0 <= next_row < self.ROWS and 0 <= next_col < self.COLUMNS):
                                break
                            
                            if self.get_piece(next_row, next_col) == player:
                                count += 1
                            else:
                                break
                        
                        if count == 4:
                            return True

    def evaluate_pattern(self, window, player):
        opponent = 2 if player == 1 else 1

        pieces_player = window.count(player)
        pieces_opponent = window.count(opponent)
        empty = window.count(0)

        def score_pattern(pieces, empty):
            if pieces == 4:
                return 1000000
            elif pieces == 3 and empty == 1:
                return 100000
            elif pieces == 2 and empty == 2:
                return 100
            return 0

        if pieces_opponent > 0 and pieces_player > 0:
            return 0

        player_score = score_pattern(pieces_player, empty)
        opponent_score = score_pattern(pieces_opponent, empty)

        return player_score - opponent_score

    def heuristic(self, player):

        if self.stop_time is not None and (self.stop_time != 0 and time.time() > self.stop_time):
            raise TimeoutError("Time limit exceeded")
            
        score = 0

        # Bonus for controlling the center column
        central_col = self.COLUMNS // 2
        for row in range(self.ROWS):
            if self.get_piece(row, central_col) == player:
                score += 10

        # Evaluating horizontal patterns
        for row in range(self.ROWS):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating vertical patterns
        for col in range(self.COLUMNS):
            for row in range(self.ROWS - 3):
                window = [self.get_piece(row + i, col) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating diagonal patterns \top-left to bottom-right\
        for row in range(self.ROWS - 3):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row + i, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        # Evaluating diagonal patterns /bottom-left to top-right/
        for row in range(self.ROWS - 3, self.ROWS):
            for col in range(self.COLUMNS - 3):
                window = [self.get_piece(row - i, col + i) for i in range(4)]
                score += self.evaluate_pattern(window, player)

        return score

# Agent function that uses Minimax to find the best move for the current player
def minimax_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1
    turn_deadline = time.time() + limit_time
    
    current_board = observation.board
    current_player = observation.mark
        
    pyplai_state = ConnectXState(current_board, current_player, turn_deadline)

    valid_moves = pyplai_state.get_moves()
    best_move = valid_moves[len(valid_moves)//2] if valid_moves else None
    
    # pyplAI >= 1.1.0 explores exactly `depth` plies (1.0.3 explored one more), so 2..5 keeps
    # the strength of the original 1..4 iterative deepening
    for depth in range(2, 6):
        try:
            
            if time.time() > turn_deadline:
                break
            
            minimax_solver = Minimax(
                ConnectXState.apply_move,
                ConnectXState.get_moves, 
                ConnectXState.is_final_state, 
                ConnectXState.wins_player, 
                ConnectXState.heuristic,
                2, 
                depth)
            
            recommended_move = minimax_solver.ejecuta(pyplai_state)

            if recommended_move is not None:
                best_move = recommended_move
            else:
                break

        except TimeoutError:
            break

    return best_move