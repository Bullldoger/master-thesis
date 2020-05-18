# Simple ODE
restart:
clc:
L := 1:
N := 25:
h := L / N:

calc_accuracy_1 := proc(N_):
    local h, accuracy:
    h := L / N_:
    exact_solution := x -> -cos(x);
    sys := {y[0] = -1, seq((y[i + 1] - y[i]) / (h) - sin(i * h) = 0, i = 0..N - 1)}:
    assign(solve(sys)):
    accuracy := evalf(sum((exact_solution(h * k) - y[k]) ** 2, k = 0..N )):
    return accuracy;
end proc:

# numerical_solution := eval(seq([k * h, y[k]], k=0..N)):
# plot([exact_solution(x), [numerical_solution]], x = 0..L);
# evalf(sum((exact_solution(h * k) - y[k]) ** 2, k = 0..N ));

calc_accuracy_1(5);