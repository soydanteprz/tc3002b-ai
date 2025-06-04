import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * Created by dhruvthakker on 01/10/21 at 9:42 am
 */
public class Season2 {
    static InputReader in = new InputReader();
    static PrintWriter out = new PrintWriter(System.out);

    public static void main(String args[]) {
        int t = in.nextInt();
        while (t-- > 0) {
            int n = in.nextInt();
            int air = in.nextInt();
            Map<Integer, Integer> tempByIndex = new HashMap<>();
            int inx[] = new int[air];
            for (int i = 0; i < air; i++) {
                int x = in.nextInt() - 1;
                tempByIndex.put(x, 0);
                inx[i] = x;
            }
            for (int i = 0; i < air; i++) {
                tempByIndex.put(inx[i], in.nextInt());
            }

            long leftMax[] = new long[n];
            long rightMax[] = new long[n];

            leftMax[0] = tempByIndex.getOrDefault(0, Integer.MAX_VALUE);
            rightMax[n - 1] = tempByIndex.getOrDefault(n - 1, Integer.MAX_VALUE);
            for (int i = 1; i < n; i++) {
                leftMax[i] = Math.min(leftMax[i - 1] + 1, tempByIndex.getOrDefault(i, Integer.MAX_VALUE));
                rightMax[n - 1 - i] = Math.min(rightMax[n - i] + 1, tempByIndex.getOrDefault(n - 1 - i, Integer.MAX_VALUE));
            }
            for (int i = 0; i < n; i++) {
                long ans = Math.min(leftMax[i], rightMax[i]);
                out.print(ans + " ");
            }
            out.println();
        }
        out.flush();
    }

    

    
}
