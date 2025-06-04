import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class C {

    static class scanner {
        static BufferedReader reader;
        static StringTokenizer tokenizer;
        static void init(InputStream input) {
            reader = new BufferedReader(new InputStreamReader(input));
            tokenizer = new StringTokenizer("");
        }
        static String next() throws IOException {
            while (!tokenizer.hasMoreTokens()) {
                tokenizer = new StringTokenizer(reader.readLine());
            }
            return tokenizer.nextToken();
        }
        static int nextInt() throws IOException {return Integer.parseInt(next());}
        
        
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    static class Pair {int first, second;public Pair(int first, int second) {this.first = first;this.second = second;}}
    
    public static void main(String[] args) throws IOException {
        scanner.init(System.in);
        int t = 1;
        t = scanner.nextInt();
        while (t-- > 0) {
            solve();
        }
    }
    
    static void solve() throws IOException {
        int n = scanner.nextInt();
        List<List<Pair>> tree = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            tree.add(new ArrayList<>());
        }
        for (int i = 0; i < n-1; i++) {
            int u = scanner.nextInt()-1;
            int v = scanner.nextInt()-1;
            tree.get(u).add(new Pair(v, i));
            tree.get(v).add(new Pair(u, i));
        }
        int start = -1;
        for (int i = 0; i < n; i++) {
            if(tree.get(i).size() > 2) {
                System.out.println(-1);
                return;
            }
            else if(tree.get(i).size() == 1) {
                start = i;
            }
        }
        int[] res = new int[n-1];

        Queue<Integer> q = new LinkedList<>();
        q.add(start);
        int weight =  2, prev = -1;
        while (!q.isEmpty()) {
            int u = q.poll();
            for(Pair v : tree.get(u)) {
                if(v.first != prev) {
                    q.add(v.first);
                    res[v.second] = weight;
                    weight = 5 - weight;
                }
            }
            prev = u;
        }

        for(int i : res) {
            System.out.print(i + " ");
        }
        System.out.println();

    }

   
}
