"""crazyJoeyv2: coordinated raiders with evidence-driven defense.

Uses only supplied observations; no background computation or external files.
UC Berkeley Pacman framework attribution: https://ai.berkeley.edu/.
Retain the framework's educational-use and non-distribution notices.
"""

from collections import Counter, deque
import heapq
import time

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red, num_training=0, **kwargs):
    """Create a symmetric team; numeric knobs are optional contest team options."""
    settings = {
        'bank': max(2, int(float(kwargs.get('bank', 6)))),
        'missing': max(4, int(float(kwargs.get('missing', 16)))),
        'guard': max(10, int(float(kwargs.get('guard', 50)))),
        'spread': max(0.0, float(kwargs.get('spread', 0.4))),
    }
    shared = TeamMemory()
    return [CrazyJoeyV2(first_index, shared, 0, settings),
            CrazyJoeyV2(second_index, shared, 1, settings)]


class TeamMemory:
    def __init__(self):
        self.beliefs = {}
        self.last_time = None
        self.last_actor = None
        self.food = None
        self.cargo = {}
        self.targets = {}
        self.missing = []
        self.missing_time = 0
        self.cargo_losses = deque(maxlen=12)

    def observe(self, agent, state):
        now = state.data.timeleft
        elapsed = 0 if self.last_time is None else max(0, self.last_time - now)
        defended = set(agent.get_food_you_are_defending(state).as_list())
        missing = (self.food or defended) - defended
        if missing:
            self.missing = sorted(missing)
            self.missing_time = now
        teammates = [state.get_agent_position(i) for i in agent.get_team(state)]
        noisy = state.get_agent_distances()
        eaters = [i for i in agent.get_opponents(state)
                  if getattr(state.get_agent_state(i), 'num_carrying', 0)
                  > self.cargo.get(i, 0)]
        for enemy in agent.get_opponents(state):
            enemy_state = state.get_agent_state(enemy)
            exact = enemy_state.get_position()
            if exact is not None:
                self.beliefs[enemy] = {nearest_point(exact): 1.0}
            else:
                prior = self.beliefs.get(enemy, {
                    state.get_initial_agent_position(enemy): 1.0})
                steps = (1 if self.last_actor is None else
                         sum((self.last_actor + k) % state.get_num_agents() == enemy
                             for k in range(elapsed)))
                # Teleporting captures/returns invalidate a motion-only prior.
                cargo = getattr(enemy_state, 'num_carrying', 0)
                if self.cargo.get(enemy, 0) > cargo and cargo == 0:
                    prior = dict(prior)
                    prior[state.get_initial_agent_position(enemy)] = 1.0
                for _ in range(min(steps, 4)):
                    predicted = Counter()
                    for pos, weight in prior.items():
                        choices = [pos] + agent.neighbors[pos]
                        for nxt in choices:
                            predicted[nxt] += weight / len(choices)
                    prior = predicted
                role = getattr(enemy_state, 'is_pacman', None)

                def compatible(pos):
                    return (all(agent.manhattan(pos, p) > 5 for p in teammates)
                            and (role is None or agent.home(pos) == role)
                            and state.get_distance_prob(
                                agent.manhattan(state.get_agent_position(agent.index), pos),
                                noisy[enemy]) > 0)

                filtered = {p: w for p, w in prior.items() if compatible(p)}
                if not filtered:
                    filtered = {p: 1.0 for p in agent.neighbors if compatible(p)}
                if len(eaters) == 1 and enemy in eaters and missing:
                    near_food = {p: w for p, w in filtered.items()
                                 if min(agent.manhattan(p, f) for f in missing) <= max(1, steps)}
                    if near_food:
                        filtered = near_food
                total = sum(filtered.values())
                self.beliefs[enemy] = ({p: w / total for p, w in filtered.items()}
                                      if total else prior)
            self.cargo[enemy] = getattr(enemy_state, 'num_carrying', 0)
        self.food = defended
        self.last_time = now
        self.last_actor = agent.index


class CrazyJoeyV2(CaptureAgent):
    def __init__(self, index, shared=None, lane=0, settings=None):
        super().__init__(index)
        self.shared = shared if shared is not None else TeamMemory()
        self.lane = lane
        self.settings = settings or {'bank': 4, 'missing': 16,
                                     'guard': 50, 'spread': 1.4}
        self.mode = 'forage'
        self.target = None
        self.recent = deque(maxlen=8)
        self.intercept_recent = deque(maxlen=16)
        self.intercept_distances = deque(maxlen=16)
        self.intercept_steps = 0
        self.last_intercept_target = None
        self.intercept_cooldown = 0
        self.defense_evidence = None
        self.last_cargo = 0
        self.last_returned = 0
        self.last_was_pacman = False
        self.risk_multiplier = 1.0

    def register_initial_state(self, state):
        super().register_initial_state(state)
        walls = state.get_walls()
        self.mid = walls.width // 2
        self.height = walls.height
        self.start = state.get_initial_agent_position(self.index)
        legal = set(walls.as_list(False))
        self.neighbors = {p: [q for q in ((p[0]+1,p[1]), (p[0]-1,p[1]),
                                        (p[0],p[1]+1), (p[0],p[1]-1)) if q in legal]
                          for p in legal}
        home_x = self.mid - 1 if self.red else self.mid
        enemy_x = self.mid if self.red else self.mid - 1
        self.gates = [(home_x,y) for y in range(walls.height)
                      if (home_x,y) in legal and (enemy_x,y) in legal]
        if not self.gates:
            self.gates = [self.start]
        self.enemy_gates = [(enemy_x,p[1]) for p in self.gates
                            if (enemy_x,p[1]) in legal]
        self.pockets = self.find_pockets()
        self.teammate = next(i for i in self.get_team(state) if i != self.index)
        self.initial_food = len(self.get_food(state).as_list())
        self.mode = 'forage'
        self.target = None
        self.recent.clear()
        self.intercept_recent.clear()
        self.intercept_distances.clear()
        self.intercept_steps = 0
        self.last_intercept_target = None
        self.intercept_cooldown = 0
        self.last_cargo = 0
        self.last_returned = 0
        self.last_was_pacman = False

    @staticmethod
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def home(self, pos):
        return pos[0] < self.mid if self.red else pos[0] >= self.mid

    def distance(self, a, b):
        return self.get_maze_distance(a, b)

    def find_pockets(self):
        """Find articulation-separated enemy regions with no home crossing."""
        discovery, low, cuts = {}, {}, set()
        parent, children = {}, Counter()
        # Iterative Tarjan traversal also supports large or corridor-heavy maps.
        for root in self.neighbors:
            if root in discovery:
                continue
            parent[root] = None
            discovery[root] = low[root] = len(discovery)
            stack = [(root, iter(self.neighbors[root]))]
            while stack:
                pos, adjacent = stack[-1]
                nxt = next(adjacent, None)
                if nxt is None:
                    stack.pop()
                    ancestor = parent[pos]
                    if ancestor is None:
                        if children[pos] > 1:
                            cuts.add(pos)
                    else:
                        low[ancestor] = min(low[ancestor], low[pos])
                        if parent[ancestor] is not None and low[pos] >= discovery[ancestor]:
                            cuts.add(ancestor)
                elif nxt != parent[pos]:
                    if nxt not in discovery:
                        parent[nxt] = pos
                        children[pos] += 1
                        discovery[nxt] = low[nxt] = len(discovery)
                        stack.append((nxt, iter(self.neighbors[nxt])))
                    else:
                        low[pos] = min(low[pos], discovery[nxt])
        pockets = {}
        for cut in sorted(cuts):
            if self.home(cut):
                continue
            seen = {cut}
            for first in self.neighbors[cut]:
                if first in seen:
                    continue
                component, todo = set(), [first]
                while todo:
                    pos = todo.pop()
                    if pos in seen:
                        continue
                    seen.add(pos)
                    component.add(pos)
                    todo.extend(self.neighbors[pos])
                if (component and not any(self.home(p) for p in component)
                        and not component.intersection(self.enemy_gates)):
                    for pos in component:
                        depth = self.distance(pos, cut)
                        # Keep the outer exit: the entire pocket must be escaped.
                        if pos not in pockets or depth > pockets[pos][1]:
                            pockets[pos] = (cut, depth)
        return pockets

    def threats(self, state):
        visible, uncertain = [], []
        for i in self.get_opponents(state):
            enemy = state.get_agent_state(i)
            if getattr(enemy, 'is_pacman', False):
                continue
            pos = enemy.get_position()
            timer = getattr(enemy, 'scared_timer', 0)
            if pos is not None:
                visible.append((nearest_point(pos), timer, i))
            else:
                belief = self.shared.beliefs.get(i, {})
                if belief and len(belief) <= 18:
                    likely = sorted(belief, key=belief.get, reverse=True)[:6]
                    uncertain.extend((p, timer, belief[p]) for p in likely)
        return visible, uncertain

    def risk(self, pos, visible, uncertain, step=1):
        if self.home(pos):
            return 0.0
        cost = 0.0
        for ghost, timer, _ in visible:
            if timer > step + 2:
                continue
            distance = self.distance(pos, ghost)
            if distance <= 1:
                cost += 500.0
            elif distance <= 5:
                cost += 15.0 * (6-distance)**2
            if pos in self.pockets:
                gate, depth = self.pockets[pos]
                if self.distance(ghost, gate) <= depth + step + 1:
                    cost += 90.0
        for ghost, timer, weight in uncertain:
            if timer <= step + 2:
                cost += weight * 10.0 * max(0, 5-self.distance(pos, ghost))**2
        return cost

    def route_action(self, state, target, visible, uncertain, deadline):
        """Bounded A*: physical route costs include ghost and pocket risk."""
        start = nearest_point(state.get_agent_position(self.index))
        legal = state.get_legal_actions(self.index)
        if start == target:
            return Directions.STOP if Directions.STOP in legal else legal[0]
        direction = {(1,0):Directions.EAST, (-1,0):Directions.WEST,
                     (0,1):Directions.NORTH, (0,-1):Directions.SOUTH}
        frontier = [(self.distance(start,target), 0.0, 0, start, None)]
        best = {start: 0.0}
        serial = 0
        fallback = None
        expanded = 0
        while frontier and expanded < 900 and time.perf_counter() < deadline:
            _, cost, _, pos, first = heapq.heappop(frontier)
            if cost > best.get(pos, float('inf')):
                continue
            if pos == target:
                return first
            expanded += 1
            for nxt in self.neighbors[pos]:
                action = first or direction[(nxt[0]-pos[0], nxt[1]-pos[1])]
                if action not in legal:
                    continue
                penalty = self.risk(nxt, visible, uncertain,
                                    min(8, int(self.distance(start,nxt))+1))
                new_cost = cost + 1 + penalty / 25.0
                if pos == start:
                    new_cost += 0.35 * self.recent.count(nxt)
                if new_cost < best.get(nxt, float('inf')):
                    best[nxt] = new_cost
                    serial += 1
                    priority = new_cost + self.distance(nxt,target)
                    heapq.heappush(frontier,(priority,new_cost,serial,nxt,action))
                    if fallback is None or priority < fallback[0]:
                        fallback = (priority, action)
        return fallback[1] if fallback else legal[0]

    def assigned_to_target(self, state, pos, location):
        """Return whether this agent is the cheaper available responder."""
        teammate = state.get_agent_state(self.teammate)
        mate_pos = nearest_point(teammate.get_position())
        me = state.get_agent_state(self.index)
        if me.num_carrying or me.scared_timer > 2:
            return False
        my_cost = (self.distance(pos, location) + (6 if me.is_pacman else 0)
                   + 4 * me.num_carrying)
        mate_cost = (self.distance(mate_pos, location)
                     + 6 * teammate.num_carrying
                     + (6 if teammate.is_pacman else 0)
                     + (30 if teammate.scared_timer > 2 else 0))
        return (my_cost, self.index) <= (mate_cost, self.teammate)

    def defense_target(self, state, pos, lead, turns_left):
        """Assign one interceptor using sightings, stolen food, and game state."""
        self.defense_evidence = None
        teammate = state.get_agent_state(self.teammate)
        mate_pos = nearest_point(teammate.get_position())
        me = state.get_agent_state(self.index)
        if me.num_carrying or me.scared_timer > 2:
            return None
        threats = []
        exact_threat = False
        for i in self.get_opponents(state):
            enemy = state.get_agent_state(i)
            if not getattr(enemy, 'is_pacman', False):
                continue
            location = enemy.get_position()
            if location is not None:
                exact_threat = True
                threats.append((nearest_point(location), getattr(enemy,'num_carrying',0)))
            else:
                belief = self.shared.beliefs.get(i, {})
                if belief and len(belief) <= 12:
                    threats.append((max(belief,key=belief.get),getattr(enemy,'num_carrying',0)))
        if not threats:
            if self.intercept_cooldown:
                return None
            # A food disappearance is exact evidence even when the invader's
            # noisy-distance belief remains broad.  The original agent stored
            # this evidence but never reacted to it.
            age = self.shared.missing_time - state.data.timeleft
            if self.shared.missing and 0 <= age <= self.settings['missing']:
                missing = sorted(self.shared.missing)
                location = min(
                    missing,
                    key=lambda p: (min(self.distance(pos, p),
                                      self.distance(mate_pos, p)), p),
                )
                if self.assigned_to_target(state, pos, location):
                    self.defense_evidence = 'missing'
                    return location
                return None

            # Protect a lead near the end instead of sending both agents on a
            # final low-value raid.  The guard window grows with the lead.
            guard_window = self.settings['guard'] + 2 * max(0, lead)
            if lead > 0 and turns_left <= guard_window:
                defended = self.get_food_you_are_defending(state).as_list()
                anchor = min(
                    self.gates,
                    key=lambda g: (
                        sum(sorted(self.distance(g, food) for food in defended)[:3])
                        if defended else 0,
                        g,
                    ),
                )
                if self.assigned_to_target(state, pos, anchor):
                    self.defense_evidence = 'guard'
                    return anchor
                return None
            return None
        location, cargo = max(threats,key=lambda t: 4*t[1]-self.distance(pos,t[0]))
        my_cost = self.distance(pos,location) + (5 if me.is_pacman else 0)
        mate_cost = (self.distance(mate_pos,location)
                     + 5*teammate.num_carrying
                     + (5 if teammate.is_pacman else 0)
                     + (30 if teammate.scared_timer > 2 else 0))
        if (my_cost, self.index) > (mate_cost,self.teammate):
            return None
        if me.is_pacman and my_cost > 12 and cargo < 4:
            return None
        # If the invader is close, capture it; otherwise cut off its return.
        if self.distance(pos,location) <= 3:
            self.defense_evidence = 'visible' if exact_threat else 'belief'
            return location
        intercepts = [g for g in self.gates
                      if self.distance(pos,g) <= self.distance(location,g)+1]
        if cargo >= 2 and intercepts:
            self.defense_evidence = 'gate'
            return min(intercepts,key=lambda g:self.distance(location,g)+0.4*self.distance(pos,g))
        self.defense_evidence = 'visible' if exact_threat else 'belief'
        return location

    def choose_action(self, state):
        deadline = time.perf_counter() + 0.30
        self.shared.observe(self, state)
        me = state.get_agent_state(self.index)
        pos = nearest_point(me.get_position())
        now = state.data.timeleft
        if (self.last_cargo > me.num_carrying
                and me.num_returned <= self.last_returned
                and self.last_was_pacman):
            self.shared.cargo_losses.append(
                (now, self.index, self.last_cargo-me.num_carrying))
        recent_losses = sum(0 <= lost_at-now <= 240 and loser == self.index
                            for lost_at, loser, _ in self.shared.cargo_losses)
        self.risk_multiplier = 1.0
        if self.intercept_cooldown:
            self.intercept_cooldown -= 1
        self.recent.append(pos)
        food = self.get_food(state).as_list()
        capsules = self.get_capsules(state)
        visible, uncertain = self.threats(state)
        home_distance = min(self.distance(pos,g) for g in self.gates)
        active_distance = min((self.distance(pos,g) for g,t,_ in visible if t <= 3),default=999)
        turns_left = max(0,(state.data.timeleft-1)//state.get_num_agents())
        safe_window = min((t for _,t,_ in visible),default=0)
        bank_limit = (self.settings['bank'] + 2
                      if safe_window > home_distance + 10
                      else self.settings['bank'])
        if active_distance <= 8 and bank_limit > 4:
            bank_limit = 4
        if recent_losses >= 2:
            bank_limit = min(bank_limit, 3)
        lead = self.get_score(state)
        winning_load = sum(state.get_agent_state(i).num_returned for i in self.get_team(state)) + me.num_carrying
        if not me.is_pacman or not me.num_carrying:
            self.mode = 'forage'
        if me.num_carrying and (me.num_carrying >= bank_limit
                or len(food) <= 2 or turns_left <= home_distance+5
                or active_distance <= 5
                or winning_load >= self.initial_food-2
                or (lead > 0 and turns_left <= self.settings['guard'])
                or (turns_left < 35 and lead <= 0 < lead+me.num_carrying)):
            self.mode = 'return'

        target = None
        if self.mode == 'return':
            other_target = self.shared.targets.get(self.teammate)
            target = min(self.gates,key=lambda g:self.distance(pos,g)
                         + 3*sum(max(0,6-self.distance(g,e)) for e,t,_ in visible
                                 if t <= home_distance+2)
                         + 2*sum(weight*max(0,5-self.distance(g,e))
                                 for e,t,weight in uncertain if t <= home_distance+2))
        else:
            target = self.defense_target(state, pos, lead, turns_left)
            if target is not None:
                self.intercept_steps += 1
                self.intercept_recent.append(pos)
                self.intercept_distances.append(self.distance(pos, target))
                stuck = (self.intercept_steps >= 12
                         and len(set(self.intercept_recent)) <= 3)
                distances = list(self.intercept_distances)
                split = len(distances) // 2
                no_progress = (len(distances) >= 12
                               and min(distances[split:]) >= min(distances[:split]))
                stalled_chase = (stuck and no_progress
                                  and self.defense_evidence != 'gate')
                overdue_inference = (self.intercept_steps >= 28
                                     and self.defense_evidence in
                                     ('belief', 'missing', 'guard'))
                if stalled_chase or overdue_inference:
                    target = None
                    self.intercept_cooldown = 8
                    self.intercept_steps = 0
                    self.intercept_recent.clear()
                    self.intercept_distances.clear()
                    self.last_intercept_target = None
                else:
                    self.mode = 'intercept'
                    self.last_intercept_target = target
            if target is None:
                self.intercept_steps = 0
                self.intercept_recent.clear()
                self.intercept_distances.clear()
                self.last_intercept_target = None
                self.mode = 'forage'
            if target is None:
                other_target = self.shared.targets.get(self.teammate)
                options = food + capsules
                def utility(p):
                    distance = self.distance(pos,p)
                    value = distance + self.risk(p,visible,uncertain, min(8,distance+1))/35.0
                    if (p[1] >= self.height/2.0) != (self.lane == 0):
                        value += 4.0
                    if p == other_target:
                        value += 7
                    if other_target is not None:
                        value += self.settings['spread'] * max(
                            0, 6-self.distance(p, other_target))
                    if p in self.pockets:
                        gate, depth = self.pockets[p]
                        blockers = [self.distance(e,gate) for e,t,_ in visible
                                    if t <= distance+depth+3]
                        if blockers and min(blockers) <= distance+depth+1:
                            value += 18 + 2*depth
                    if p in capsules:
                        if active_distance <= 6 or (me.num_carrying and safe_window < home_distance+5):
                            value -= 9
                        elif safe_window > distance+12:
                            value += 15
                        else:
                            value -= 2
                        mate = state.get_agent_state(self.teammate)
                        if mate.is_pacman and safe_window < 5:
                            value -= 2 + min(5,mate.num_carrying)
                    if p == self.target:
                        value -= 1.5
                    # Avoid needless round trips for a lone remote food dot.
                    if p in food:
                        value -= 0.3*sum(self.distance(p,f)<=3 for f in food)
                    return value, p
                if options:
                    target = min(options,key=utility)
                else:
                    target = min(self.gates,key=lambda g:self.distance(pos,g))
        self.target = target
        self.shared.targets[self.index] = target
        action = self.route_action(state,target,visible,uncertain,deadline)

        # Check immediate enemy replies: an adjacent ghost may act before us.
        choices = []
        for candidate in state.get_legal_actions(self.index):
            successor = state.generate_successor(self.index,candidate)
            nxt = successor.get_agent_state(self.index)
            next_pos = nearest_point(nxt.get_position())
            gain = self.get_score(successor)-lead
            dead = next_pos == self.start and pos != self.start and gain <= 0
            successor_visible, successor_uncertain = self.threats(successor)
            danger = self.risk(next_pos,successor_visible,successor_uncertain)
            if nxt.is_pacman:
                for enemy_pos,timer,enemy_index in successor_visible:
                    if timer <= 1 and self.distance(next_pos,enemy_pos)<=1:
                        danger += 2000
            value = -danger - (5000+100*me.num_carrying if dead else 0)
            value += 2000*gain
            value -= 3*self.distance(next_pos,target)
            if not (self.mode == 'intercept' and next_pos == target):
                value -= 4*self.recent.count(next_pos)
            if candidate == action:
                value += 15
            if candidate == Directions.STOP and pos != target:
                value -= 6
            choices.append((value,candidate))
        selected = max(choices,key=lambda item:item[0])[1]
        self.last_cargo = me.num_carrying
        self.last_returned = me.num_returned
        self.last_was_pacman = me.is_pacman
        return selected

# Candidate version 5
