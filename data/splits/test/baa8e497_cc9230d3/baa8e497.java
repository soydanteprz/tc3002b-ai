import java.util.*;
import java.io.*;

public class Solution {
    public static void main(String str[]) throws IOException{
        Reader sc = new Reader();
        int t = sc.nextInt();
        while(t-->0) {
            int n = sc.nextInt();
            ArrayList<Tree> tt = new ArrayList<>();
            for(int i=0;i<n;i++){
                Tree temp = new Tree(sc.nextInt(), sc.nextInt(), i);
                tt.add(temp);
            }
            for(int i=0;i<n-1;i++){
                int x = sc.nextInt()-1;
                int y = sc.nextInt()-1;
                tt.get(x).al.add(tt.get(y));
                tt.get(y).al.add(tt.get(x));
            }
            Pair p = traversal(tt.get(0), null);
            System.out.println(Math.max(p.av, p.bv));
        }
    }
    static Pair traversal(Tree node, Tree Parent){
        if(node.al.size()==1 && Parent!=null){
            Pair p = new Pair(0,0);
            return p;
        }
        long lt = 0;
        long rt = 0;
        for(Tree t: node.al){
            if(t!=Parent){
                Pair p = traversal(t, node);
                long ll = Math.abs(node.l- t.l) + p.av;
                ll = Math.max(ll, Math.abs(node.l-t.r)+p.bv);
                long rr = Math.abs(node.r- t.l) + p.av;
                rr = Math.max(rr, Math.abs(node.r-t.r)+p.bv);
                lt+=ll;
                rt+=rr;
            }
        }
        Pair p = new Pair(lt, rt);
        return p;
    }
    

}
class Pair{
    long av;
    long bv;
    Pair(long av, long bv){
        this.av = av;
        this.bv = bv;
    }

}
class Tree{
    int l;
    int r;
    int i;
    ArrayList<Tree> al = new ArrayList<>();
    Tree(int l, int r, int i){
        this.l = l;
        this.r = r;
        this.i = i;
    }

}
