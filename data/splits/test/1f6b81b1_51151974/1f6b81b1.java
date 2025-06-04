import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;


public class InterestingStory {

    public static void main(String[] args) {

        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader sc = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Solver solver = new Solver();
		int t = sc.nextInt();
//        int t = 1;
        while (t-- != 0) {
            solver.solve(sc, out);
        }
        out.close();

    }

    static class Solver {
        public void solve(InputReader sc, PrintWriter out) {
            int n = sc.nextInt();
            char[][] words = new char[n][];
            for(int i = 0; i < n; i++) words[i] = sc.next().toCharArray();
            int[][] arr = new int[n][];
            for(int i = 0; i < n; i++) {
                arr[i] = new int[words[i].length];
                for(int j = 0; j < arr[i].length; j++) arr[i][j] = words[i][j]-'a';
            }
            int max = 0;
            for(int now = 0; now < 5; now++) {
                PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
                for(int i = 0; i < n; i++) {
                    int nowcnt = 0;
                    int othercnt = 0;
                    for(int j = 0; j < arr[i].length; j++) {
                        if(arr[i][j]==now) nowcnt++;
                        else othercnt++;
                    }
                    pq.add(nowcnt-othercnt);
                }
                int canTake = 0;
                int sum = 0;
                while(!pq.isEmpty()) {
                    int nowAdd = pq.poll();
                    if(sum+nowAdd>0) {
                        sum += nowAdd;
                        canTake++;
                    } else {
                        break;
                    }
                }
                max = Math.max(max,canTake);
            }
            out.println(max);
        }

    }

    

    

    

    

    


}
