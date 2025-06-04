import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.StringTokenizer;

public class D {

   private static final String INPUT_FILE_PATH = "";

   int n, a, b, da, db;

   ArrayList<Integer>[] adj;
   int[] depth;
   boolean[] visited;
   int diameter;

   int dfs(int u, int parent, int currDepth) {
      visited[u] = true;
      depth[u] = currDepth;

      ArrayList<Integer> childDepths = new ArrayList();
      for (int v : adj[u]) {
         if (v != parent && !visited[v]) {
            int childDepth = dfs(v, u, currDepth + 1);
            childDepths.add(childDepth);
         }
      }
      Collections.sort(childDepths);

      // Update diameter
      int n = childDepths.size();
      if (n >= 2) {
         int d1 = childDepths.get(n - 1) - currDepth;
         int d2 = childDepths.get(n - 2) - currDepth;
         diameter = Math.max(diameter, d1 + d2);
      } else if (n == 1) {
         diameter = Math.max(diameter, childDepths.get(n - 1) - currDepth);
      } else {
         diameter = Math.max(diameter, 1);
      }

      return childDepths.isEmpty() ? currDepth : childDepths.get(n - 1);
   }

   void solve() {
      int t = in.nextInt();
      while (t-- > 0) {
         n = in.nextInt();
         a = in.nextInt();
         b = in.nextInt();
         da = in.nextInt();
         db = in.nextInt();
         adj = new ArrayList[n + 1];
         for (int i = 1; i <= n; i++) adj[i] = new ArrayList();
         for (int i = 0; i < n - 1; i++) {
            int u = in.nextInt(), v = in.nextInt();
            adj[u].add(v);
            adj[v].add(u);
         }

         depth = new int[n + 1];
         visited = new boolean[n + 1];
         diameter = 0;

         dfs(a, -1, 0);

         if (depth[b] <= da
            || 2 * da >= db
            || 2 * da >= diameter) {
            out.println("Alice");
         } else {
            out.println("Bob");
         }
      }
   }

   private final InputReader in;
   private final PrintWriter out;

   private D(InputReader in, PrintWriter out) {
      this.in = in;
      this.out = out;
   }

   

   public static void main(String[] args) throws Exception {
      InputStream inputStream = INPUT_FILE_PATH.isEmpty()
         ? System.in
         : new FileInputStream(new File(INPUT_FILE_PATH));
      OutputStream outputStream = System.out;

      InputReader inputReader = new InputReader(inputStream);
      PrintWriter printWriter = new PrintWriter(outputStream);

      new D(inputReader, printWriter).solve();

      printWriter.close();
   }

}
