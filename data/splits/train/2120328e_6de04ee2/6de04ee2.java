import java.util.*;
public class Solution{
    public static void main(String[] args){
    Scanner sc=new Scanner(System.in);
    int t=sc.nextInt();

    while(t-->0) {
        int n=sc.nextInt();
        int[] a=new int[n];
        for(int i=0;i<n;i++)
             a[i]=sc.nextInt();

     char[] c=sc.next().toCharArray();

        Vector<Integer> l=new Vector<>(), r=new Vector<>();
        for(int i=0;i<n;i++)
            (c[i] == 'B' ? l : r).add(a[i]);
        Collections.sort(l);
        Collections.sort(r,Collections.reverseOrder());

        boolean ok = true;
        for(int i=0;i<l.size();i++)
            if (l.get(i) < i + 1)
                ok = false;
        for(int i=0;i<r.size();i++)
            if (r.get(i) > n - i)
                ok = false;

         System.out.print((ok ? "YES" : "NO")+'\n');
    }
    }
}
