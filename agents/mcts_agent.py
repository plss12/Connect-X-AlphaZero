# Kaggle submissions cannot install pyplAI, so the MCTS algorithm is vendored below verbatim from
# pyplAI v1.1.0 (https://github.com/plss12/pyplAI/blob/v1.1.0/pyplAI/Algoritmos.py).
# tests/test_pyplai_sync.py checks that it still matches the installed pyplAI.
from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

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


def _generador(semilla: Optional[int]) -> Any:
    #Sin semilla se usa el módulo random, así random.seed() sigue controlando la biblioteca
    return random if semilla is None else random.Random(semilla)


def _comprueba_movimientos(movimientos: list, estado: Any) -> None:
    if len(movimientos) == 0:
        raise ValueError(
            "El estado no es final pero no tiene movimientos posibles. Todo estado no final "
            "debe tener al menos un movimiento (estado: " + repr(estado) + ")")


def _recompensa(gana_jugador: Callable, estado: Any, jugadores: List[int]) -> List[float]:
    #Vector de recompensas: 1 para cada ganador, 0 para el resto y 0.5 para todos si hay empate
    res = [1.0 if gana_jugador(estado, jugador) else 0.0 for jugador in jugadores]
    if 1.0 not in res:
        res = [0.5] * len(jugadores)
    return res


class _Presupuesto:
    """Controla el límite de la búsqueda: un tiempo en segundos o un número fijo de iteraciones."""

    def __init__(self, tiempoEjecucion: float, iteraciones: Optional[int]):
        if iteraciones is not None and iteraciones < 1:
            raise ValueError("iteraciones debe ser un entero mayor que 0")
        self.tiempoEjecucion = tiempoEjecucion
        self.iteraciones = iteraciones
        self.t0 = time.perf_counter()
        self.realizadas = 0

    def inicia(self) -> None:
        self.t0 = time.perf_counter()
        self.realizadas = 0

    def continua(self) -> bool:
        #Se llama tras cada iteración, así siempre se hace al menos una y la raíz tiene algún hijo
        self.realizadas += 1
        if self.iteraciones is not None:
            return self.realizadas < self.iteraciones
        return time.perf_counter() - self.t0 < self.tiempoEjecucion

    def transcurrido(self) -> float:
        return time.perf_counter() - self.t0


class MCTS:
    """Monte Carlo Tree Search con UCT (algoritmo 2 de Browne et al., 2012).

    Las recompensas son un vector con una entrada por jugador y cada nodo se elige con la
    recompensa del jugador que mueve en su padre, lo que permite más de dos jugadores.
    """
    k = 1/math.sqrt(2) #Coeficiente de exploración Cp

    def __init__(self, aplicar_movimiento, obtener_movimientos, es_estado_final, gana_jugador, numeroJugadores, tiempoEjecucion, estadisticas=False, *, iteraciones=None, semilla=None, seleccion_final="media"):
        if seleccion_final not in ("media", "visitas"):
            raise ValueError("seleccion_final debe ser 'media' o 'visitas'")
        self.aplicar_movimiento = aplicar_movimiento
        self.obtener_movimientos = obtener_movimientos
        self.es_estado_final = es_estado_final
        self.gana_jugador = gana_jugador
        self.jugadores = [i+1 for i in range(numeroJugadores)]
        self.tiempoEjecucion = tiempoEjecucion
        self.estadisticas = estadisticas
        self.seleccion_final = seleccion_final
        self.presupuesto = _Presupuesto(tiempoEjecucion, iteraciones)
        self.rng = _generador(semilla)

    class nodo:
        def __init__(self,padre,mov=None):
            self.mov=mov #Movimiento que ha generado el nodo
            self.n = 0 #Número de veces que se ha visitado el nodo
            self.q = [] #Vector de recompensas acumuladas
            self.hijos = [] #Lista de nodos hijos
            self.padre = padre #Nodo padre

    @staticmethod
    def movs_restantes(v,movs):
        res=list(movs)
        for hijo in v.hijos:
            if(hijo.mov in res):
                res.remove(hijo.mov)
        return res

    @staticmethod
    def nodos_creados(v):
        suma = 1
        pendientes = list(v.hijos)
        while pendientes:
            hijo = pendientes.pop()
            suma += 1
            pendientes.extend(hijo.hijos)
        return suma

    def busca(self, s0):
        """Ejecuta la búsqueda desde s0 y devuelve el nodo raíz."""
        v0 = self.nodo(None)
        v0.q = [0]*len(self.jugadores)
        movimientos = self.obtener_movimientos(s0)
        self.presupuesto.inicia()
        while True:
            v1, s1, movs = self.tree_policy(v0, s0, movimientos)
            delta = self.default_policy(s1, movs, self.jugadores)
            self.backup(v1, delta, self.jugadores)
            if not self.presupuesto.continua():
                break
        return v0

    def _ordena_hijos(self, v0, jugador):
        if self.seleccion_final == "visitas":
            #Robust child: el hijo más visitado
            return sorted(v0.hijos, key=lambda h: h.n, reverse=True)
        #BESTCHILD(v0, 0) de Browne et al.: el hijo con mayor recompensa media
        return sorted(v0.hijos, key=lambda h: h.q[jugador]/h.n, reverse=True)

    def analiza(self, s0):
        """Busca desde s0 y devuelve un ResultadoAnalisis con el ranking de movimientos. No imprime nada."""
        if self.es_estado_final(s0) or len(self.obtener_movimientos(s0)) == 0:
            return ResultadoAnalisis(None)
        v0 = self.busca(s0)
        jugador = s0.jugadorActual-1
        hijos = self._ordena_hijos(v0, jugador)
        return ResultadoAnalisis(
            movimiento=hijos[0].mov,
            movimientos=[EvaluacionMovimiento(h.mov, h.n, h.q[jugador]/h.n) for h in hijos],
            iteraciones=v0.n,
            nodos=self.nodos_creados(v0),
            tiempo=self.presupuesto.transcurrido())

    def ejecuta(self, s0):
        movimientos=self.obtener_movimientos(s0)
        if(len(movimientos)==1):
            return movimientos[0]
        elif(len(movimientos)==0 or self.es_estado_final(s0)):
            return None
        resultado = self.analiza(s0)
        if(self.estadisticas):
            print("\nTiempo de ejecución: ", resultado.tiempo)
            print("Número de nodos creados: ", resultado.nodos)
            print("Número de nodos visitados: ", resultado.iteraciones)
        return resultado.movimiento

    def tree_policy(self, v,s,movimientos):
        while(self.es_estado_final(s)==False):
            _comprueba_movimientos(movimientos, s)
            movs_sin_visitar = self.movs_restantes(v,movimientos)
            if(0<len(movs_sin_visitar)):
                return self.expand(v,s,movs_sin_visitar)
            else:
                jugador=s.jugadorActual-1
                mejorHijo = self.best_child(v, MCTS.k ,jugador)
                copiaEstado=deepcopy(s)
                s = self.aplicar_movimiento(copiaEstado,mejorHijo.mov)
                movimientos = self.obtener_movimientos(s)
                v = mejorHijo
        return [v,s,movimientos]

    def expand(self, v,s,movRestantes):
        copiaEstado=deepcopy(s)
        mov = self.rng.choice(movRestantes)
        s = self.aplicar_movimiento(copiaEstado, mov)
        movs = self.obtener_movimientos(s)
        hijo = self.nodo(v,mov)
        hijo.q = [0]*len(v.q)
        v.hijos.append(hijo)
        return [hijo,s,movs]

    @staticmethod
    def best_child(v,c,jugador):
        #UCT: Q(v')/N(v') + c * sqrt(2 ln N(v) / N(v'))
        mejorHijo = None
        maximo = -math.inf
        for hijo in v.hijos:
            heuristica = hijo.q[jugador]/hijo.n + (c * math.sqrt((2*math.log(v.n))/hijo.n))
            if(heuristica>maximo):
                maximo = heuristica
                mejorHijo = hijo
        return mejorHijo

    def default_policy(self, s,movs,jugadores):
        while(self.es_estado_final(s)==False):
            _comprueba_movimientos(movs, s)
            a = self.rng.choice(movs)
            s = self.aplicar_movimiento(s,a)
            movs = self.obtener_movimientos(s)
        return _recompensa(self.gana_jugador, s, jugadores)

    @staticmethod
    def backup(v,delta,jugadores):
        while(v is not None):
            v.n = v.n+1
            for i in range(len(jugadores)):
                v.q[i]+=delta[i]
            v = v.padre


# Adapting ConnectX to pyplAI style
class ConnectXState:
    ROWS = 6
    COLUMNS = 7

    def __init__(self, board_data, current_player=1):
        self.board = list(board_data)
        self.jugadorActual = current_player

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


# Agent function that uses MCTS to find the best move for the current player
def mcts_agent(observation, configuration):
    limit_time = configuration.timeout - 0.1
    current_board = observation.board
    current_player = observation.mark
    
    pyplai_state = ConnectXState(current_board, current_player)
    
    mcts_solver = MCTS(
         ConnectXState.apply_move,
         ConnectXState.get_moves, 
         ConnectXState.is_final_state, 
         ConnectXState.wins_player, 
         2, 
         limit_time,
         estadisticas=True)

    recommended_move = mcts_solver.ejecuta(pyplai_state)
    
    return recommended_move