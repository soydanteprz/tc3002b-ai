import java.util.*;
import java.io.*;

public class D {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        
        int T = in.nextInt();
        while(T-- > 0) {
            int n = in.nextInt();
            int[] a = new int[n];
            for(int j=0;j<n;j++) a[j] = in.nextInt();
            char[] s = in.next().toCharArray();

            List<Integer> blue = new ArrayList<>();
            List<Integer> red = new ArrayList<>();
            for(int j=0;j<n;j++) {
                if(s[j] == 'B') blue.add(a[j]);
                else red.add(a[j]);
            }

            Collections.sort(blue);
            Collections.sort(red);

            boolean p = true;
            int cur = 1;
            for(int val : blue) {
                if(val<cur) {
                    p = false;
                    break;
                }
                else cur++;
            }
            for(int val : red) {
                if(val>cur) {
                    p = false;
                    break;
                }
                else cur++;
            }
            
            if(p) System.out.println("yes");
            else System.out.println("no");
        }
    }
}
