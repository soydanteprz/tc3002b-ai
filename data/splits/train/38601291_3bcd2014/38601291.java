import java.io.*;
import java.util.*;

public class stones {
    public static void main (String[] args) throws IOException {
        // set up
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        //BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream("test.in")));
        PrintWriter out = new PrintWriter(System.out);
        int T = Integer.parseInt(input.readLine());
        for (int i=0;i<T;i++) {
            // read in test case
            int n = Integer.parseInt(input.readLine());
            int[] seq = new int[n];
            StringTokenizer st = new StringTokenizer(input.readLine());
            int low = 0;
            int high = 0;
            for (int j=0;j<n;j++) {
                seq[j] = Integer.parseInt(st.nextToken());
                high = Math.max(high, seq[j]);
            }
            // binary search on answer
            while (low<high) {
                int mid = (low+high+1)/2;
                //System.out.println(possible(seq, mid));
                if (possible(seq, mid)) {
                    low = mid; }
                else high = mid-1; }
            out.println(low);
        }
        out.close();

    }

    static boolean possible(int[] seq, int goal) {
        /* checks if array can be made to have smallest value
        greater than or equal to given goal value */
        //System.out.println(goal);
        int L = seq.length;
        int[] arr = Arrays.copyOf(seq, L);
        //System.out.println(Arrays.toString(arr));
        for (int i=L-1;i>=2;i--) {
            //System.out.println(Arrays.toString(arr));
            if (arr[i] < goal) return false;
            int max_d = Math.min((arr[i] - goal)/3, seq[i]/3);
            //System.out.println(arr[i]);
            //System.out.println(max_d);
            arr[i-1] += max_d;
            arr[i-2] += max_d*2;
            arr[i] -= max_d*3;
        }
        //System.out.println(Arrays.toString(arr));
        for (int num: arr) {
            if (num < goal) return false; }
        return true;
    }
}
