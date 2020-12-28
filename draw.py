#!/usr/bin/env python
import random
import copy
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


class Particle:
    def newton(self, t, Y, q, m, B):
        # Via https://flothesof.github.io/charged-particle-trajectories-E-and-B-fields.html
        (x, y, z, u, v, w) = Y[0:6]
        q = self.cw * (self.T - t) ** (-self.alpha)
        alpha = q / m * B
        return np.array([u, v, w, 0, alpha * w, -alpha * v])

    def __init__(self, area, T, α, v, collide=True):
        self.area = area
        self.PID = 1
        # how granular?
        self.granularity_exp = 3
        self.granularity = 10 ** (self.granularity_exp - 1)

        self.collide = collide
        if self.collide:
            self.touched = np.zeros(
                (
                    int(self.area[0] * self.granularity) + 40, # safety factor so we can just blindly write larger.
                    int(self.area[1] * self.granularity) + 40,
                )
            )
            self.starts = copy.deepcopy(self.touched)

        self.particle_params = {}
        self.particle_traces = {}
        self.dt = 0.05

        self.Tarr = T
        self.αarr = α
        self.varr = v

    def throwi(self, x, y, vx, vy, optsIndex=0, **kw):
        zT = self.Tarr[optsIndex]
        zA = self.αarr[optsIndex]

        velocityWeWant = self.varr[optsIndex]
        currentVelocity = np.linalg.norm((vx, vy))
        vF = (velocityWeWant / currentVelocity)
        return self.throw(x, y, vx * vF, vy * vF, zT, zA, vF=0.3/velocityWeWant, **kw)

    def throw(self, x, y, vx, vy, zT, zA, vF=5, mustNotDie=True, cw=True):
        # print(f'throw {x:0.2f} {y:0.2f} {vx:0.2f} {vy:0.2f} {zT} {zA}')
        self.T = zT * self.dt
        self.alpha = zA
        r = ode(self.newton).set_integrator("dopri5")
        self.cw = 1 if cw else -1

        self.PID += 1
        t0 = 0
        x0 = np.array([0, x, y])
        v0 = np.array([1, vx, vy])
        initial_conditions = np.concatenate((x0, v0))
        r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 1.0, 1.0)

        positions = []
        broke = False
        new_exclusions = []
        while r.successful() and r.t < self.T - self.dt:
            r.integrate(r.t + self.dt)

            if not (0 < r.y[1] < self.area[0]):
                # print('b1')
                broke = True
                break

            if not (0 < r.y[2] < self.area[1]):
                # print('b2')
                broke = True
                break

            g_x = int(round(r.y[1], self.granularity_exp - 1) * self.granularity)
            g_y = int(round(r.y[2], self.granularity_exp - 1) * self.granularity)

            # 0.5 isn't great, just need to have some exclusion to allow it to survive exiting.
            if self.collide and r.t > vF:
                # If we're NOT in an untouched cell (0), exit
                if self.touched[g_x, g_y] != 0:
                    # print('b3', r.t, self.touched[g_x, g_y])
                    # (We can't hit our own PID because we don't add until after.)
                    broke = True
                    break

                w = 2 ** (self.granularity_exp - 2)
                for i in range(-w, w):
                    for j in range(-w, w):
                        if g_x + i > 0 and g_y + j > 0:
                            new_exclusions.append((g_x + i, g_y + j))

            positions.append(r.y)  # keeping pos (0:3) + velocity (3:)

        if mustNotDie and broke:
            # Failed
            return

        # We've committed, this oen is good to go. Record everything
        positions = np.array(positions)
        self.particle_params[self.PID] = {
            'T': zT,
            'a': zA,
            'v': np.linalg.norm((vx, vy)),
        }
        self.particle_traces[self.PID] = positions
        for (qx, qy) in new_exclusions:
            self.touched[qx, qy] = self.PID
        return self.PID

    def _touched2scatter(self, var):
        t2 = []
        for i in range(int(self.area[0] * self.granularity)):
            for j in range(int(self.area[1] * self.granularity)):
                if var[i, j]:
                    t2.append((i / self.granularity, j/self.granularity))
        t2 = np.array(t2)
        return t2

    def plot(self):
        fig, ax = plt.subplots(figsize=(self.area[0] * 2, self.area[1] * 2))

        # Plot our exclusion zone
        # if self.collide:
            # t2 = self._touched2scatter(self.touched)
            # ax.scatter(t2[:, 0], t2[:, 1], c='#ff3300ee')
            # t3 = self._touched2scatter(self.starts)
            # ax.scatter(t3[:, 0], t3[:, 1], c='#3300ffee')

        for k, trace in self.particle_traces.items():
            # print(f"Plotting {k} {self.particle_params[k]}")
            ax.plot(trace[:, 1], trace[:, 2], label=f"P{k}")


        plt.xlim([0, self.area[0]])
        plt.ylim([0, self.area[1]])
        print('saving')
        # # plt.show()
        plt.savefig("draw.png")
        print('saved')

    def save(self):
        for k, trace in self.particle_traces.items():
            with open(f"out-{k}.tsv", "w") as handle:
                writer = csv.writer(handle, delimiter="\t")
                for (x, y) in zip(trace[:, 1], trace[:, 2]):
                    writer.writerow((x, y))

    def _try_smaller(self, px, pv, oIs, cw):
        for i in range(oIs, len(self.Tarr)):
            r_child = pm.throwi(*px, *pv, optsIndex=i, cw=cw)
            if r_child is not None:
                return r_child, i
        return None, None

    def _get_sps(self, r):
        positions = self.particle_traces[r]
        ox = max(len(positions) // 16, 60)
        offsets = range(0, len(positions), ox)[1:-1]
        for idx, i in enumerate(offsets):
            px = positions[i][1:3]
            pv = positions[i][4:6]
            ois = self.Tarr.index(self.particle_params[r]['T'])            # This is gross

            # logging
            g_x = int(round(px[0], self.granularity_exp - 1) * self.granularity)
            g_y = int(round(px[1], self.granularity_exp - 1) * self.granularity)
            for x in range(-2, 2):
                for y in range(-2, 2):
                    self.starts[g_x + x, g_y + y] = 1
            yield (px, pv, ois, idx % 2 == 1)

    def fill(self):
        # We know this succeeds
        r = pm.throwi(4, 0, -0.4, 0.5, optsIndex=0, cw=True, mustNotDie=False)

        # We'll keep track of places we can start + size we should start with?
        startingPoints = []
        startingPoints.extend(self._get_sps(r))
        iteration = 0
        while True:
            iteration += 1
            # If we're out of places to initiate, exit.
            if len(startingPoints) == 0:
                break

            print(f'iter={iteration} possibleStarts={len(startingPoints)}')
            if iteration > 580:
                break

            # Shuffle so we try diff places?
            random.shuffle(startingPoints)
            # random.choice
            (sppx, sppv, spois, spcw) = startingPoints[0]

            # We try and find one of the options
            (rc, ri) = self._try_smaller(sppx, sppv, oIs=spois, cw=spcw)

            # If we do
            if rc is not None:
                sps2 = list(self._get_sps(rc))
                # Add the new SPs
                startingPoints.extend(sps2)

            # Remove the old one
            # either it was used, or it is unusable.
            del startingPoints[0]
            # And carry on



pm = Particle(
    area=(9, 12),
    T=[1000, 900, 600, 400],
    α=[.8, .6, .5, .4],
    v=[0.256, 0.128, 0.34, 0.12],
    # collide=False
)

pm.fill()
pm.plot()
# pm.save()
