import java.util.*;
import java.lang.*;
import java.io.*;

public class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        int sm, n;
        while(t > 0) {
            t--;
            n = sc.nextInt();
            String s1,s2;
            s1 = sc.next();
            s2 = sc.next();
            int a[] = new int[4];
            a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0;
            for(int i = 0 ; i < n ; i++) {
                if(s1.charAt(i) == '0'&& s2.charAt(i) == '1') a[0]++;
                else if(s1.charAt(i) == '1'&& s2.charAt(i) == '0') a[1]++;
                else if(s1.charAt(i) == '1'&& s2.charAt(i) == '1') a[2]++;
                else a[3]++;
            }
            // System.out.println(a[0] + " " + a[1] + " " + a[2] + " " + a[3]);
            int n1 = Integer.MAX_VALUE, n2 = Integer.MAX_VALUE, n3 = Integer.MAX_VALUE;
            
            if (a[0] == a[1]) {
                n1 = 2*a[0];
            }
            if((a[2] - 1) == a[3]) {
                // System.out.println(a[3] + 1);
                n2 = 2*a[3] + 1;
            } 
            if((a[3] + 1) == a[2]) {
                // System.out.println(a[2] + 1);
                n3 = 2*a[2] + 1;
            } 
            int ans = Math.min(n1, Math.min(n2,n3));
            if(ans == Integer.MAX_VALUE) {
                System.out.println("-1");
            } else {
                System.out.println(ans);
            }
        }
	}
}
