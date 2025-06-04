import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class Main {
    static int MOD = 1000000007;

    // After writing solution, quick scan for:
    //   array out of bounds
    //   special cases e.g. n=1?
    //
    // Big numbers arithmetic bugs:
    //   int overflow
    //   sorting, or taking max, after MOD
    void solve() throws IOException {
        int n = ri();
        int[] h = ril(n);
        
        List<List<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        int[] nextGe = new int[n];
        nextGe[n-1] = n;
        for (int i = n-2; i >= 0; i--) {
            int j = i+1;
            while (j != n && h[j] < h[i]) j = nextGe[j];
            nextGe[i] = j;
        }
        int[] nextLe = new int[n];
        nextLe[n-1] = n;
        for (int i = n-2; i >= 0; i--) {
            int j = i+1;
            while (j != n && h[j] > h[i]) j = nextLe[j];
            nextLe[i] = j;
        }
        int[] prevGe = new int[n];
        prevGe[0] = -1;
        for (int i = 1; i < n; i++) {
            int j = i-1;
            while (j != -1 && h[j] < h[i]) j = prevGe[j];
            prevGe[i] = j;
        }
        int[] prevLe = new int[n];
        prevLe[0] = -1;
        for (int i = 1; i < n; i++) {
            int j = i-1;
            while (j != -1 && h[j] > h[i]) j = prevLe[j];
            prevLe[i] = j;
        }
        for (int i = 0; i < n; i++) {
            if (prevLe[i] != -1) adj.get(prevLe[i]).add(i);
            if (prevGe[i] != -1) adj.get(prevGe[i]).add(i);
            if (nextLe[i] != n) adj.get(i).add(nextLe[i]);
            if (nextGe[i] != n) adj.get(i).add(nextGe[i]);
        }

        int dist = 0;
        Deque<Integer> q = new ArrayDeque<>();
        boolean[] visited = new boolean[n];
        q.addLast(0);
        visited[0] = true;
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int u = q.removeFirst();
                if (u == n-1) {
                    q.clear();
                    break;
                }
                for (int v : adj.get(u)) {
                    if (!visited[v]) {
                        visited[v] = true;
                        q.addLast(v);
                    }
                }
            }
            dist++;
        }
        pw.println(dist-1);
    }

    // Template code below

    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    PrintWriter pw = new PrintWriter(System.out);

    public static void main(String[] args) throws IOException {
        Main m = new Main();
        m.solve();
        m.close();
    }

    void close() throws IOException {
        pw.flush();
        pw.close();
        br.close();
    }

    int ri() throws IOException {
        return Integer.parseInt(br.readLine());
    }

    

    int[] ril(int n) throws IOException {
        int[] nums = new int[n];
        int c = 0;
        for (int i = 0; i < n; i++) {
            int sign = 1;
            c = br.read();
            int x = 0;
            if (c == '-') {
                sign = -1;
                c = br.read();
            }
            while (c >= '0' && c <= '9') {
                x = x * 10 + c - '0';
                c = br.read();
            }
            nums[i] = x * sign;
        }
        while (c != '\n' && c != -1) c = br.read();
        return nums;
    }

    

    

    

    

    
}
