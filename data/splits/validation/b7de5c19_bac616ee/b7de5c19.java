import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int t = scan.nextInt();
        for (int i = 0; i < t; i++) {
            int n = scan.nextInt();
            ArrayList<ArrayList<Pair>> graph = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                graph.add(new ArrayList<>());
            }
            for (int j = 0; j < n - 1; j++) {
                int u;
                int v;
                u = scan.nextInt();
                v = scan.nextInt();
                u--;
                v--;
                graph.get(u).add(new Pair(v, j));
                graph.get(v).add(new Pair(u, j));
            }
            boolean soluble = true;
            int curV = 0;
            int prevV = -1;
            int[] ans = new int[n];
            int prime = 2;
            for (int j = 0; j < n; j++) {
                ArrayList<Pair> list = graph.get(j);
                if (list.size() > 2) {
                    soluble = false;
                } else if (list.size() == 1) {
                    curV = j;
                }
            }
            if (soluble) {
                for (int j = 0; j < n - 1; j++) {
                    ArrayList<Pair> list = graph.get(curV);
                    for (int z = 0; z < list.size(); z++) {
                        if (list.get(z).vertex != prevV) {
                            ans[list.get(z).numberOfEdge] = prime;
                            prime = changePrime(prime);
                            prevV = curV;
                            curV = list.get(z).vertex;
                            break;
                        }
                    }
                }

                for (int j = 0; j < n - 1; j++) {
                    System.out.print(ans[j] + " ");
                }
                System.out.println();
            } else {
                System.out.println(-1);
            }
        }
    }
    public static int changePrime(int prime) {
        if (prime == 2) {
            prime = 3;
        } else {
            prime = 2;
        }
        return prime;
    }
}

class Pair {
    int vertex;
    int numberOfEdge;

    public Pair(int vertex, int numberOfEdge) {
        this.vertex = vertex;
        this.numberOfEdge = numberOfEdge;
    }
}
