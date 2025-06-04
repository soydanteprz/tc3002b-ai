import java.io.*;
import java.util.*;

import static java.lang.Double.parseDouble;
import static java.lang.Integer.compare;
import static java.lang.Integer.parseInt;
import static java.lang.Long.parseLong;
import static java.lang.System.in;
import static java.lang.System.out;
import static java.util.Arrays.asList;
import static java.util.Collections.max;
import static java.util.Collections.min;

public class Main {
    private static final int MOD = (int) (1E9 + 7);
    private static final int INF = (int) (1E9);
    static FastScanner scanner = new FastScanner(in);

    public static void main(String[] args) throws IOException {
        // Write your solution here
        StringBuilder answer = new StringBuilder();
        int t = 1;
        t = parseInt(scanner.nextLine());
        while (t-- > 0) {
            answer.append(solve()).append("\n");
        }
//        out.println(answer);
    }

    private static Object solve() throws IOException {
       int n = scanner.nextInt();
       int m = scanner.nextInt();
       int x = scanner.nextInt();
       Integer[] h = new Integer[n];
       for (int i = 0; i < n; i++) {
          h[i] = scanner.nextInt();
       }
       TreeSet<Pair> q = new TreeSet<>();
       PriorityQueue<Pair> height = new PriorityQueue<>();
       for (int i = 0; i < m; i++) {
          q.add(new Pair(i + 1,0));
       }
       for (int i = 0; i < n; i++) {
          height.add(new Pair(i,h[i]));
       }
       boolean shift = false;
       int size = 0;
       while(!height.isEmpty()){
          Pair p = height.poll();
          Pair building  = (shift) ? q.pollLast() : q.pollFirst();
          h[p.index] = building.index;
          q.add(new Pair(building.index,building.value + p.value));
           //out.println("q = " + q);
          if(++size == n){
              size = 0;
              shift = !shift;
          }
       }
       if(safe(q,x)){
          out.println("YES");
          StringBuilder stringBuilder = new StringBuilder();
          for (Object o : h)
             stringBuilder.append(o).append(" ");
          out.println(stringBuilder);
       }
       else {
           out.println("NO");
       }
        return "";
    }

   private static boolean safe(TreeSet<Pair> q, int x) {
       int last = q.pollFirst().value;
       while (!q.isEmpty()){
          Pair p = q.pollFirst();
          if(p.value - last > x) return false;
          last = p.value;
       }
       return true;
   }

   private static class Pair implements Comparable<Pair> {
        int index, value;

        public Pair(int index, int value) {
            this.index = index;
            this.value = value;
        }

        

       
   }

    

    

    

    

    private static int add(int a, int b) {
        long res = ((long) (a + MOD) % MOD + (b + MOD) % MOD) % MOD;
        return (int) res;
    }

    


    

    

    
}
