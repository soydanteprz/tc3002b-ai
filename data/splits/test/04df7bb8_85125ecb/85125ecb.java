import java.io.*;
import java.util.Arrays;
import java.util.Hashtable;

public class AirConditionersUpsolve{
    

    // region variables
    static InputReader sc = new InputReader();
    static OutputStream outputStream = System.out;
    static PrintWriter w = new PrintWriter(outputStream);
    // endregion

    private static void initiateIO()
            throws IOException {if (System.getProperty("ONLINE_JUDGE") == null) { try { w = new PrintWriter("output.txt");sc = new InputReader(new FileInputStream("input.txt")); } catch (Exception e) { throw new IOException(); }} }

    public static void main(String[] args)
            throws IOException {

        initiateIO();

        int t = sc.nextInt();
        while(t-- > 0) {
            solve();
        }

        w.close();

    }

    static void solve() throws IOException {
        int n = sc.nextInt();
        int k = sc.nextInt();
        int[] ac = new int[n+1];
        Arrays.fill(ac, Integer.MAX_VALUE);
        int[] pos = new int[k+1];
        for(int i = 1; i <= k; i++) {
            pos[i] = sc.nextInt();
        }
        for(int i = 1; i <= k; i++) {
            int posi = pos[i];
            int temp = sc.nextInt();
            ac[posi] = temp;
        }

        long[] prev = new long[n+1];

        long temp = Integer.MAX_VALUE;
        for(int i = 1; i <= n; i++) {
            temp = Math.min(temp+1, ac[i]);
            prev[i] = temp;
        }

        long[] next = new long[n+1];

        temp = Integer.MAX_VALUE;
        for(int i = n; i > 0; i--) {
            temp = Math.min(temp+1, ac[i]);
            next[i] = temp;
        }

        for(int i = 1; i <= n; i++) {
            w.print(Math.min(prev[i], next[i])+" ");
        }
        w.println();
    }
}
