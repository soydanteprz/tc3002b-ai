import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        for (int i = 0; i < t; i++) {
            notSitting(scanner);
        }
    }

    private static void notSitting(Scanner scanner) {
        int n = scanner.nextInt();
        int[] num = new int[n - 1];
        int[] ct = new int[n];
        boolean[] visited = new boolean[n];
        List<int[]>[] edges = new List[n];
        for (int i = 0; i < n; i++) {
            edges[i] = new ArrayList<>();
        }
        for (int i = 0; i < n - 1; i++) {
            int u = scanner.nextInt() - 1;
            int v = scanner.nextInt() - 1;
            edges[u].add(new int[]{v, i});
            edges[v].add(new int[]{u, i});
            ct[u]++;
            ct[v]++;
        }
        int start = -1;
        for (int i = 0; i < n; i++) {
            if (ct[i] >= 3) {
                System.out.println(-1);
                return;
            } else if (ct[i] == 1) {
                start = i;
            }
        }
        visited[start] = true;
        boolean odd = false;
        for (int i = 0; i < n - 1; i++) {
            for (int[] e : edges[start]) {
                if (!visited[e[0]]) {
                    visited[e[0]] = true;
                    num[e[1]] = odd ? 3 : 2;
                    odd = !odd;
                    start = e[0];
                    break;
                }
            }
        }
        for (int i = 0; i < n - 1; i++) {
            System.out.print(num[i]);
            System.out.print(" ");
        }
        System.out.println();
    }
}
